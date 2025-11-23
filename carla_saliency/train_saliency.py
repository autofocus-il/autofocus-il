#!/usr/bin/env python3

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import hydra
from omegaconf import DictConfig

from data_utils import GazePreprocessor
from data_utils.gaze_utils import get_gaze_mask, apply_gmd_dropout
from models import BCActor, Encoder, VectorQuantizer, weight_init
from train.common.base_trainer import BaseTrainer
from train.common.data import build_dataloader
from train.common.distributed import wrap_ddp, destroy_distributed_if_initialized
from saliency_dataloader import build_obs_specs, build_dataset

class SaliencyTrainer(BaseTrainer):
    """Trainer class for behavior cloning with gaze regularization"""
    
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.gaze_key = self.cfg.data.gaze_key
        self.gaze_preprocessor = GazePreprocessor(
            img_height=self.cfg.data.img_height,
            img_width=self.cfg.data.img_width,
            gaze_sigma=self.cfg.gaze.mask_sigma,
            maxpoints=self.cfg.gaze.max_points,
            device=str(self.device),
            temporal_alpha = float(self.cfg.gaze.temporal_alpha),
            temporal_beta = float(self.cfg.gaze.temporal_beta),
            temporal_gamma = float(self.cfg.gaze.temporal_gamma),
            temporal_use_future = bool(self.cfg.gaze.temporal_use_future),
            temporal_window = getattr(self.cfg.gaze, "temporal_window", None),
        )
        self.criterion = nn.MSELoss()
        
    def _setup_data(self):
        # Initialize obs specs with configured gaze key
        build_obs_specs(self.cfg.data)
        dataset = build_dataset(self.cfg.data, gaze_ratio=float(self.cfg.gaze.ratio)) # torch Dataset
        self._print_rank0("Dataset info:")
        self._print_rank0(f"  Total sequences: {len(dataset)}")
        self._print_rank0(f"  Number of demos: {dataset.n_demos}")
        self._print_rank0(f"  Cache mode: {self.cfg.data.cache_mode}")
        self._print_rank0(f"  Frame stack: {self.cfg.data.frame_stack}")
        self._print_rank0(f"  Gaze key: {self.cfg.data.gaze_key}")

        sampler = None
        if self.is_distributed:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=False)
        self.train_sampler = sampler
        self.train_dataset = dataset
        self.train_loader = build_dataloader(
            dataset,
            self.cfg.data,
            sampler,
            grad_accum_steps=self.cfg.training.gradient_accumulation_steps,
        )
        self._print_rank0(f"Train dataloader created with {len(self.train_loader)} batches")    
        
    def build_models(self):
        model_cfg = self.cfg.model
        
        extra_channels = model_cfg.frame_stack if self.cfg.gaze.method == 'ViSaRL' else 0
        input_channels = extra_channels + model_cfg.frame_stack * (1 if model_cfg.grayscale else 3)
        
        self.encoder = Encoder(
            input_channels=input_channels,
            embedding_dim=model_cfg.embedding_dim,
            num_hiddens=model_cfg.num_hiddens,
            num_residual_layers=model_cfg.num_residual_layers,
            num_residual_hiddens=model_cfg.num_residual_hiddens,
        ).to(self.device)
        
        encoder_output_dim = model_cfg.last_feature_map_dim[0] * model_cfg.last_feature_map_dim[1] * model_cfg.embedding_dim
        
        self.actor = BCActor(
            encoder_output_dim=encoder_output_dim,
            latent_dim=model_cfg.z_dim,
            action_dim=self.cfg.data.action_dim,
        ).to(self.device)
        
        # weight init
        self.encoder.apply(weight_init)
        
        # AGIL encoder
        self.encoder_agil = None
        if self.cfg.gaze.method == 'AGIL':
            self.encoder_agil = Encoder(
                input_channels=input_channels,
                embedding_dim=model_cfg.embedding_dim,
                num_hiddens=model_cfg.num_hiddens,
                num_residual_layers=model_cfg.num_residual_layers,
                num_residual_hiddens=model_cfg.num_residual_hiddens,
            ).to(self.device)
            self.encoder_agil.apply(weight_init)
        
        # GRIL gaze coordinate predictor
        self.gril_gaze_coord_predictor = None
        if self.cfg.gaze.method == 'GRIL':
            self.gril_gaze_coord_predictor = nn.Sequential(nn.Linear(model_cfg.z_dim, model_cfg.z_dim), nn.ReLU(), nn.Linear(model_cfg.z_dim, self.cfg.gaze.max_points * 2)).to(self.device)
            self.gril_gaze_coord_predictor.apply(weight_init)
        
        # OREO quantizer
        self.quantizer = None
        if self.cfg.dropout.method == 'Oreo':
            self.quantizer = VectorQuantizer(model_cfg.embedding_dim, self.cfg.dropout.num_embeddings, 0.25).to(self.device)
            vqvae_path = Path(self.cfg.dropout.vqvae_path)
            if vqvae_path.exists():
                for p in self.quantizer.parameters():
                    p.requires_grad = False
                vqvae_dict = torch.load(vqvae_path, map_location="cpu", weights_only=True)
                self.encoder.load_state_dict({k[9:]: v for k, v in vqvae_dict.items() if "_encoder" in k})
                self.quantizer.load_state_dict({k[11:]: v for k, v in vqvae_dict.items() if "_quantizer" in k})
                self._print_rank0(f"Loaded VQ-VAE from {vqvae_path}")
            else:
                self._print_rank0(f"Warning: VQ-VAE model not found at {vqvae_path}")
        
        # Model Compile
        if self.cfg.training.use_compile:
            self._print_rank0("Compiling models...")
            self.encoder = torch.compile(self.encoder)
            self.actor = torch.compile(self.actor)
            if self.encoder_agil is not None:
                self.encoder_agil = torch.compile(self.encoder_agil)
            if self.gril_gaze_coord_predictor is not None:
                self.gril_gaze_coord_predictor = torch.compile(self.gril_gaze_coord_predictor)

        # DDP Wrap
        if self.is_distributed:
            self.encoder = wrap_ddp(self.encoder, self.cfg.training, self.local_rank)
            self.actor = wrap_ddp(self.actor, self.cfg.training, self.local_rank)
            if self.encoder_agil is not None:
                self.encoder_agil = wrap_ddp(self.encoder_agil, self.cfg.training, self.local_rank)
            if self.gril_gaze_coord_predictor is not None:
                self.gril_gaze_coord_predictor = wrap_ddp(self.gril_gaze_coord_predictor, self.cfg.training, self.local_rank)

    def get_optim_params(self):
        params = list(self.encoder.parameters()) + list(self.actor.parameters())
        if self.encoder_agil is not None:
            params += list(self.encoder_agil.parameters())
        if self.gril_gaze_coord_predictor is not None:
            params += list(self.gril_gaze_coord_predictor.parameters())
        return params
    
    def compute_gaze_regularization_loss(
        self,
        feature_tensor: torch.Tensor,
        gaze_heatmaps: torch.Tensor,
        gaze_coords: torch.Tensor,
        obs_images: torch.Tensor,
        is_valid_gaze: torch.Tensor
    ) -> torch.Tensor:
        """Compute gaze regularization loss based on the configured method"""
        
        reg_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        
        if self.cfg.gaze.method in ['Teacher', 'Reg']:
            # Gaze regularization loss
            with torch.no_grad():
                g1 = gaze_heatmaps[:, -1:, :, :][is_valid_gaze > 0].float()
            
            g2 = get_gaze_mask(feature_tensor, self.cfg.gaze.beta, (obs_images.shape[-2], obs_images.shape[-1]))[is_valid_gaze > 0]
            
            if self.cfg.gaze.prob_dist_type in ['TV', 'JS', 'KL']:
                # Normalize to probability distributions
                g1_sum = g1.sum(dim=(-1, -2, -3), keepdim=True) + 1e-8
                g2_sum = g2.sum(dim=(-1, -2, -3), keepdim=True) + 1e-8
                g1 = g1 / g1_sum.detach()
                g2 = g2 / g2_sum.detach()
            
            def KL(a, b):
                return (a * torch.log((a + 1e-6) / (b + 1e-6))).sum(dim=(1,2,3)).mean(0)
            
            if self.cfg.gaze.prob_dist_type == 'KL':
                reg_loss = KL(g1, g2)
            elif self.cfg.gaze.prob_dist_type == 'TV':
                reg_loss = (g1 - g2).abs().sum(dim=(1,2,3)).mean(0)
            elif self.cfg.gaze.prob_dist_type == 'JS':
                reg_loss = 0.5 * (KL(g1, (g1+g2)/2) + KL(g2, (g1+g2)/2))
            elif self.cfg.gaze.prob_dist_type == 'MSE':
                reg_loss = F.mse_loss(g1, g2)
            else:
                raise ValueError(f'Invalid prob_dist_type: {self.cfg.gaze.prob_dist_type}')
        
        elif self.cfg.gaze.method == 'Contrastive':
            positive_images = gaze_heatmaps[is_valid_gaze > 0][:, :self.cfg.data.frame_stack]
            negative_images = gaze_heatmaps[is_valid_gaze > 0][:, self.cfg.data.frame_stack:]
            z_plus = self.encoder(positive_images)
            z_minus = self.encoder(negative_images)
            t1 = torch.linalg.vector_norm(feature_tensor[is_valid_gaze > 0] - z_plus, dim=(1, 2, 3)) ** 2
            t2 = torch.linalg.vector_norm(feature_tensor[is_valid_gaze > 0] - z_minus, dim=(1, 2, 3)) ** 2
            reg_loss = torch.max(torch.zeros_like(t1), t1 - t2 + self.cfg.gaze.contrastive_threshold).mean()
        
        elif self.cfg.gaze.method == 'GRIL' and self.gril_gaze_coord_predictor is not None:
            if is_valid_gaze.sum() > 0:
                # For GRIL, the feature tensor is the flattened latent from the actor head
                gaze_coord_pred = self.gril_gaze_coord_predictor(feature_tensor[is_valid_gaze > 0])
                gaze_coords_valid = gaze_coords[is_valid_gaze > 0].float()
                if gaze_coords_valid.dim() == 3:
                    gaze_target = rearrange(gaze_coords_valid, 'b p c -> b (p c)')
                else:
                    gaze_target = gaze_coords_valid
                gaze_coord_loss = F.mse_loss(gaze_coord_pred, gaze_target) + 1e-8
                reg_loss = torch.clamp(gaze_coord_loss, min=0.0, max=100.0)
        
        return reg_loss
    
    def set_train_mode(self):
        self.encoder.train()
        self.actor.train()
        if self.encoder_agil is not None:
            self.encoder_agil.train()
        if self.gril_gaze_coord_predictor is not None:
            self.gril_gaze_coord_predictor.train()

    def compute_loss(self, batch):
        """
        Called by BaseTrainer.train()
        """
        obs_image = batch['obs']['image'].to(self.device, non_blocking=True)
        gaze_coords_raw = batch['obs'][self.gaze_key].to(self.device, non_blocking=True)
        actions = batch['actions'].to(self.device, non_blocking=True)

        # get temporal aggregated saliency heatmaps
        obs_image, gaze_heatmaps, center_idx = self.gaze_preprocessor.prepare_for_bc(
            obs_image_seq=obs_image,
            gaze_seq=gaze_coords_raw,
            frame_stack=self.cfg.data.frame_stack,
            grayscale=self.cfg.model.grayscale,
            aggregate_stack=bool(self.cfg.gaze.temporal_flag),
        )
        # align actions with center_idx
        if actions.dim() == 3:
            # if considering both past and future
            if center_idx < actions.shape[1]:
                actions = actions[:, center_idx, :]
            # if only considering past
            else:
                actions = actions[:, -1, :]
        batch_size = obs_image.shape[0]
        is_valid_gaze = batch['is_valid_gaze'].to(self.device, non_blocking=True).view(-1)
        with torch.amp.autocast('cuda', enabled=self.cfg.training.use_amp):
            # For GRIL, use coordinates at center timestep if sequence provided
            if gaze_coords_raw.dim() >= 3:
                gc = gaze_coords_raw[:, center_idx]
            else:
                gc = gaze_coords_raw
            # Build encoder input per gaze method to avoid channel mismatch
            gaze_mask = is_valid_gaze.view(-1, 1, 1, 1).expand_as(gaze_heatmaps)
            overlay = obs_image * gaze_heatmaps
            overlay_min = overlay.amin(dim=(1, 2, 3), keepdim=True)
            overlay_max = overlay.amax(dim=(1, 2, 3), keepdim=True)
            overlay = (overlay - overlay_min) / (overlay_max - overlay_min + 1e-8)
            
            # heatmap overlaied obs or original obs
            masked_obs_heatmap_overlay = torch.where(
                                        gaze_mask > 0,
                                        overlay,
                                        obs_image
                                    )
            
            # masked gaze heatmaps (whole black masked or gaze heatmap)
            masked_gaze_heatmaps = gaze_mask * gaze_heatmaps
            
            if self.cfg.gaze.method == 'Mask':
                    enc_in = masked_obs_heatmap_overlay
            elif self.cfg.gaze.method == 'ViSaRL':
                # Concatenate image stack with gaze heatmaps along channel dim
                enc_in = torch.cat([obs_image, masked_gaze_heatmaps], dim=1)
            else:
                enc_in = obs_image
                
            gaze_dropout_mask = masked_gaze_heatmaps if self.cfg.dropout.method == 'IGMD' else None
            z = self.encoder(enc_in, dropout_mask=gaze_dropout_mask)
            
            if self.cfg.gaze.method == 'AGIL' and self.encoder_agil is not None:
                obs_or_heatmap = torch.where(
                            gaze_mask > 0,
                            gaze_heatmaps,
                            obs_image
                        )
                z_agil = self.encoder_agil(obs_or_heatmap)
                z = 0.5 * (z + z_agil)
            if self.cfg.dropout.method == 'GMD':
                    z = apply_gmd_dropout(z, masked_gaze_heatmaps)
            elif self.cfg.dropout.method == 'Oreo' and self.quantizer is not None:
                with torch.no_grad():
                    _, _, _, _, encoding_indices, _ = self.quantizer(z)
                    prob = torch.ones(batch_size * self.cfg.dropout.oreo_num_mask, self.cfg.dropout.num_embeddings) * (1 - self.cfg.dropout.oreo_prob)
                    code_mask = torch.bernoulli(prob).to(self.device)
                    encoding_indices_flatten = rearrange(encoding_indices, 'b h w -> (b h w)')
                    encoding_indices_onehot = torch.zeros((len(encoding_indices_flatten), self.cfg.dropout.num_embeddings), device=encoding_indices_flatten.device)
                    encoding_indices_onehot.scatter_(1, encoding_indices_flatten.unsqueeze(1), 1)
                    encoding_indices_onehot = rearrange(encoding_indices_onehot, '(b hw) e -> b hw e', b=batch_size)
                    mask = (code_mask.unsqueeze(1) * repeat(encoding_indices_onehot, 'b hw e -> (m b) hw e', m=self.cfg.dropout.oreo_num_mask)).sum(2)
                    mask = rearrange(mask, 'b (h w) -> b h w', h=20, w=38)
                mask = rearrange(mask, 'b h w -> b 1 h w')
                z = repeat(z, 'b c h w -> (m b) c h w', m=self.cfg.dropout.oreo_num_mask) * mask
                z = z / (1.0 - self.cfg.dropout.oreo_prob)
                actions = repeat(actions, 'b a -> (m b) a', m=self.cfg.dropout.oreo_num_mask)
            
            pred_actions, actor_latent = self.actor(z, return_latent=True)
            actor_loss = self.criterion(pred_actions, actions)
            
            if self.cfg.gaze.method == 'GRIL' and self.gril_gaze_coord_predictor is not None:
                reg_loss = self.compute_gaze_regularization_loss(actor_latent, gaze_heatmaps, gc, obs_image, is_valid_gaze)
            else:
                reg_loss = self.compute_gaze_regularization_loss(z, gaze_heatmaps, gc, obs_image, is_valid_gaze)
            total = self.cfg.gaze.lambda_weight * reg_loss + actor_loss
        metrics = {"Loss/actor": float(actor_loss.item()), "Loss/reg": float(reg_loss.item())}
        return total, batch_size, metrics

    def save_for_epoch(self, epoch: int):
        """
        Called by BaseTrainer.train()
        """
        enc_to_save = self.encoder.module if hasattr(self.encoder, 'module') else self.encoder
        act_to_save = self.actor.module if hasattr(self.actor, 'module') else self.actor
        torch.save(enc_to_save.state_dict(), self.checkpoint_dir / f"ep{epoch}_encoder.pth")
        torch.save(act_to_save.state_dict(), self.checkpoint_dir / f"ep{epoch}_actor.pth")
        if self.gril_gaze_coord_predictor is not None:
            gril_to_save = self.gril_gaze_coord_predictor.module if hasattr(self.gril_gaze_coord_predictor, 'module') else self.gril_gaze_coord_predictor
            torch.save(gril_to_save.state_dict(), self.checkpoint_dir / f"ep{epoch}_gril_gaze_coord_predictor.pth")
        if self.encoder_agil is not None:
            agil_to_save = self.encoder_agil.module if hasattr(self.encoder_agil, 'module') else self.encoder_agil
            torch.save(agil_to_save.state_dict(), self.checkpoint_dir / f"ep{epoch}_encoder_agil.pth")
        if self.cfg.logging.save_params:
            # Save params expected by eval/my_agents/bc_agent.py
            params = {
                'gaze_method': self.cfg.gaze.method,
                'dp_method': self.cfg.dropout.method,
                'grayscale': self.cfg.model.grayscale,
                'stack': self.cfg.model.frame_stack,
                'embedding_dim': self.cfg.model.embedding_dim,
                'num_embeddings': self.cfg.dropout.num_embeddings,
                'num_hiddens': self.cfg.model.num_hiddens,
                'num_residual_layers': self.cfg.model.num_residual_layers,
                'num_residual_hiddens': self.cfg.model.num_residual_hiddens,
                'z_dim': self.cfg.model.z_dim,
                # Path for optional gaze predictor used by some eval agents
                'gaze_predictor_path': self.cfg.gaze.gaze_predictor_path,
                'models_path': self.checkpoint_dir,
                'epochs': epoch,
                'action_dim': self.cfg.data.action_dim,
            }
            self.experiment.save_params_json(params)
    
@hydra.main(version_base=None, config_path="./configs", config_name="train_saliency_base")
def main(cfg: DictConfig):
    """Main entry point"""
    import signal, sys

    def handle_sigint(sig, frame):
        print("SIGINT received. Cleaning up DDP...")
        destroy_distributed_if_initialized()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        trainer = SaliencyTrainer(cfg)
        trainer.train()
        trainer._print_rank0("Training completed!")
    except KeyboardInterrupt:
        print("Training interrupted by user! Cleaning up...")
    finally:
        destroy_distributed_if_initialized()


if __name__ == "__main__":
    main()
