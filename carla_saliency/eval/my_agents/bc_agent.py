#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a human agent to control the ego vehicle via keyboard
"""
from gc import enable
import torch
import torch.nn as nn
import numpy as np
import cv2
import json
import os
import yaml
from my_agents.autonomous_agent import (
    AutonomousAgent,
    control_to_vector,
    vector_to_control,
    noop_control,
)
from models.linear_models import BCActor, Encoder, Decoder, AutoEncoder


def apply_gmd_dropout(z, g, test_mode=False):
    dropout_prob = 0.7
    B, C, H, W = z.shape
    A = torch.rand([B, 1, H, W], device=z.device)
    K = torch.nn.functional.interpolate(g, size=(H, W), mode="bicubic")
    K = (K - K.min()) / (K.max() - K.min())
    K = dropout_prob * K + (1 - dropout_prob)
    M = (A < K).float()
    if test_mode:
        z = z * K
    else:
        z = z * M
    return z


class BCAgent(AutonomousAgent):
    """
    BC agent to control the ego vehicle using the trained model
    """

    def setup(self, args):
        """
        Setup the agent parameters
        """
        super(BCAgent, self).setup(args)
        with open(args.params_path + "/params.json") as f:
            params = json.load(f)

        self.gaze_method = params["gaze_method"]
        self.dp_method = params["dp_method"]
        self.grayscale = params["grayscale"]
        self.stack = params["stack"]
        self.embedding_dim = params["embedding_dim"]
        self.num_embeddings = params["num_embeddings"]
        self.num_hiddens = params["num_hiddens"]
        self.num_residual_layers = params["num_residual_layers"]
        self.num_residual_hiddens = params["num_residual_hiddens"]
        self.z_dim = params["z_dim"]
        self.gaze_predictor_path = params["gaze_predictor_path"]
        self.models_path = params["models_path"]
        self.epochs = params["epochs"]
        self.action_dim = params["action_dim"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # whether to overlay confounded indicators on recorded frames
        self.confounded = getattr(args, "confounded", False)
        self.confounded_cfg = None
        self.prev_control = None
        if self.confounded:
            try:
                with open(
                    getattr(
                        args,
                        "confounded_config",
                        "carla_saliency/configs/confounded_render.yaml",
                    ),
                    "r",
                ) as yf:
                    self.confounded_cfg = yaml.safe_load(yf)
            except Exception:
                self.confounded_cfg = {
                    "render": {
                        "anchor": "top_mid",
                        "top_mid_offset_y": 14,
                        "dot": {
                            "enabled": True,
                            "color_bgr": [0, 0, 255],
                            "radius": 7,
                            "margin_top": 10,
                            "offset_x": 0,
                        },
                        "arrow": {
                            "enabled": True,
                            "color_bgr": [255, 255, 255],
                            "gap_from_dot": 8,
                            "gap_left": 28,
                            "gap_right": 10,
                            "y_offset": 0,
                            "length": 32,
                            "thickness": 2,
                            "head_size": 6,
                        },
                    }
                }

        if self.raw_files != "":
            self.z_list = []
            self.encoder_input_list = []

        if self.gaze_method in ["ViSaRL", "Mask", "AGIL"] or self.dp_method in [
            "GMD",
            "IGMD",
        ]:
            encoder_gp = Encoder(
                self.stack * (1 if self.grayscale else 3),
                self.embedding_dim,
                self.num_hiddens,
                self.num_residual_layers,
                self.num_residual_hiddens,
            )
            decoder_gp = Decoder(
                self.stack * (1 if self.grayscale else 3),
                self.embedding_dim,
                self.num_hiddens,
                self.num_residual_layers,
                self.num_residual_hiddens,
            )
            self.gaze_predictor = AutoEncoder(encoder_gp, decoder_gp).to(self.device)
            # Load state dict with prefix removal
            state_dict = torch.load(self.gaze_predictor_path, weights_only=True)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("_orig_mod.", "").replace("module.", "")
                new_state_dict[new_key] = v
            self.gaze_predictor.load_state_dict(new_state_dict)
            self.gaze_predictor.eval()

        coeff = 2 if self.gaze_method == "ViSaRL" else 1
        self.encoder = Encoder(
            coeff * self.stack * (1 if self.grayscale else 3),
            self.embedding_dim,
            self.num_hiddens,
            self.num_residual_layers,
            self.num_residual_hiddens,
        ).to(self.device)
        encoder_output_dim = 20 * 38 * self.embedding_dim
        self.actor = BCActor(encoder_output_dim, self.z_dim, self.action_dim).to(
            self.device
        )
        self.encoder_agil = None
        if self.gaze_method == "AGIL":
            self.encoder_agil = Encoder(
                self.stack * (1 if self.grayscale else 3),
                self.embedding_dim,
                self.num_hiddens,
                self.num_residual_layers,
                self.num_residual_hiddens,
            ).to(self.device)

        # Load encoder with prefix removal
        state_dict = torch.load(
            self.models_path + "/ep{}_encoder.pth".format(self.epochs),
            weights_only=True,
        )
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "").replace("module.", "")
            new_state_dict[new_key] = v
        self.encoder.load_state_dict(new_state_dict)

        # Load actor with prefix removal
        state_dict = torch.load(
            self.models_path + "/ep{}_actor.pth".format(self.epochs), weights_only=True
        )
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "").replace("module.", "")
            new_state_dict[new_key] = v
        self.actor.load_state_dict(new_state_dict)

        if self.gaze_method == "AGIL":
            # Load encoder_agil with prefix removal
            state_dict = torch.load(
                self.models_path + "/ep{}_encoder_agil.pth".format(self.epochs),
                weights_only=True,
            )
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("_orig_mod.", "").replace("module.", "")
                new_state_dict[new_key] = v
            self.encoder_agil.load_state_dict(new_state_dict)

        self.encoder.eval()
        self.actor.eval()
        if self.gaze_method == "AGIL":
            self.encoder_agil.eval()

        self.frames_stack = []
        self.debug_enabled = True  # Control debug output switch

        print("Params are:", params)
        print("                ")
        # exit()

    def _debug_print_step_info(
        self,
        step,
        input_data,
        image_center,
        obs,
        stacked_obs_np,
        stacked_obs_torch,
        final_model_input,
        gaze_heatmap=None,
        action_vector=None,
        control=None,
    ):
        """Debug print function, manages all debug output unifiedly"""
        if not self.debug_enabled:
            return

        # First 5 steps: Image processing pipeline check
        if step < 5:
            print(f"\n=== Step {step} Debug Info ===")
            print(f"Raw CARLA image shape: {input_data['Center'][1].shape}")
            print(f"After [:, :, -2::-1] shape: {image_center.shape}")
            print(
                f"Image dtype: {image_center.dtype}, range: [{image_center.min()}, {image_center.max()}]"
            )
            print(
                f"Resolution settings: render={self.render_res_w}x{self.render_res_h}, obs={self.obs_res_w}x{self.obs_res_h}x{self.obs_res_c}"
            )
            print(f"Model config: grayscale={self.grayscale}, stack={self.stack}")
            print(f"Gaze method: {self.gaze_method}, DP method: {self.dp_method}")
            print(f"Confounded enabled: {self.confounded}")

            if self.obs_res_c == 1:
                print(
                    f"Converted to grayscale: {obs.shape}, range: [{obs.min()}, {obs.max()}]"
                )

            if (
                self.obs_res_w != self.render_res_w
                or self.obs_res_h != self.render_res_h
            ):
                print(f"Resized obs: {obs.shape}")

            print(f"Stacked obs shape: {stacked_obs_np.shape}")
            print(f"After permute (0,3,1,2): {stacked_obs_torch.shape}")

            if self.grayscale:
                if stacked_obs_torch.shape[1] == 3:
                    print(f"Applying weighted grayscale conversion (RGB channels)")
                else:
                    print(f"Already single channel, skipping grayscale conversion")

            print(f"Final model input shape: {final_model_input.shape}")
            print(
                f"Model input range: [{final_model_input.min().item():.4f}, {final_model_input.max().item():.4f}]"
            )

            if gaze_heatmap is not None:
                print(
                    f"Gaze heatmap shape: {gaze_heatmap.shape}, range: [{gaze_heatmap.min().item():.4f}, {gaze_heatmap.max().item():.4f}]"
                )

                if self.gaze_method == "ViSaRL":
                    print(f"Applied ViSaRL concat")
                elif self.gaze_method == "Mask":
                    print(f"Applied gaze mask")
                elif self.gaze_method == "AGIL":
                    print(f"Applied AGIL ensemble")

                if self.dp_method == "IGMD":
                    print(f"IGMD dropout mask applied")
                elif self.dp_method == "GMD":
                    print(f"Applied GMD dropout")

        # First 30 steps: Action output diagnostics
        if step < 30 and action_vector is not None and control is not None:
            print(f"Step {step} - Raw action: {action_vector}")
            print(
                f"  throttle={action_vector[0]:.4f}, steer={action_vector[1]:.4f}, brake={action_vector[2]:.4f}"
            )
            print(
                f"  handbrake={action_vector[3]:.4f}, reverse={action_vector[4]:.4f}, manual_gear={action_vector[5]:.4f}, gear={action_vector[6]:.4f}"
            )
            print(
                f"  -> Control: throttle={control.throttle:.4f}, steer={control.steer:.4f}, brake={control.brake}"
            )
            if control.brake:
                print(f"  *** BRAKING! (brake={control.brake}) ***")

    def _debug_print_overlay_info(self, step, success=True, error=None):
        """Debug print overlay related info"""
        if not self.debug_enabled or step >= 5:
            return
        if success:
            print(
                f"Applied confounded overlay for both model input and video recording"
            )
        else:
            print(f"Overlay failed: {error}")

    def _debug_print_encoder_info(self, step, z_shape):
        """Debug print encoder output info"""
        if not self.debug_enabled or step >= 5:
            return
        print(f"Encoder output shape: {z_shape}")

    def _process_observation(self, image_rgb):
        """Process RGB image into model input format."""
        obs = image_rgb.copy()
        if self.obs_res_c == 1:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        if self.obs_res_w != self.render_res_w or self.obs_res_h != self.render_res_h:
            obs = cv2.resize(obs, (self.obs_res_w, self.obs_res_h))

        # Update frame stack
        if len(self.frames_stack) == 0:
            for _ in range(self.stack):
                self.frames_stack.append(obs.copy())
        else:
            self.frames_stack.append(obs)
            self.frames_stack.pop(0)

        # Convert to tensor format matching training: [B, S, H, W, C] then to [B, S*C, H, W]
        stacked_obs_np = np.stack(self.frames_stack, axis=0)  # [S, H, W, C]

        # Add batch dimension to match training format
        stacked_obs_np = stacked_obs_np[np.newaxis, ...]  # [1, S, H, W, C]
        stacked_obs_torch = torch.from_numpy(stacked_obs_np.copy())

        # Apply same processing as training (_format_obs_image)
        if stacked_obs_torch.dtype == torch.uint8:
            stacked_obs_torch = stacked_obs_torch.float() / 255.0

        # [B, S, H, W, C] -> [B, S, C, H, W]
        stacked_obs_torch = stacked_obs_torch.permute(0, 1, 4, 2, 3)

        if self.grayscale and stacked_obs_torch.shape[2] == 3:
            # Apply RGB to grayscale conversion: 0.299*R + 0.587*G + 0.114*B
            stacked_obs_torch = (
                0.299 * stacked_obs_torch[:, :, 0:1]
                + 0.587 * stacked_obs_torch[:, :, 1:2]
                + 0.114 * stacked_obs_torch[:, :, 2:3]
            )

        # [B, S, C, H, W] -> [B, S*C, H, W] (concatenate stack frames along channel dimension)
        B, S, C, H, W = stacked_obs_torch.shape
        final_model_input = stacked_obs_torch.reshape(B, S * C, H, W).to(self.device)

        return final_model_input

    def _predict_control(self, model_input):
        """Predict control from processed model input."""
        with torch.no_grad():
            g = None
            if self.gaze_method in ["ViSaRL", "Mask", "AGIL"] or self.dp_method in [
                "GMD",
                "IGMD",
            ]:
                g = self.gaze_predictor(model_input)
                g[g < 0] = 0
                g[g > 1] = 1

            encoder_input = model_input
            if self.gaze_method == "ViSaRL":
                encoder_input = torch.cat([model_input, g], dim=1)
            elif self.gaze_method == "Mask":
                encoder_input = model_input * g
            else:
                encoder_input = model_input

            dropout_mask = None
            if self.dp_method == "IGMD":
                dropout_mask = g[:, -1:]

            z = self.encoder(encoder_input, dropout_mask=dropout_mask)

            if self.gaze_method == "AGIL":
                z = (z + self.encoder_agil(model_input * g)) / 2

            if self.dp_method == "GMD":
                z = apply_gmd_dropout(z, g[:, -1:], test_mode=True)

            action, _ = self.actor(z, return_latent=True)

        v = action.cpu()[0].numpy()
        control = vector_to_control(v)
        return control

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.

        When confounded mode is enabled, applies overlay based on PREVIOUS action
        (stored in self.prev_control), then predicts next action with the confounded image.
        This matches the training procedure where overlay is based on the previous/current action label.
        """

        # Convert CARLA BGRA -> RGB and drop alpha
        image_center_rgb = input_data["Center"][1][:, :, -2::-1]

        # For confounded mode, apply overlay based on PREVIOUS action (like training)
        if self.confounded:
            # Apply overlay based on previous action (or noop for first step)
            try:
                if self.prev_control is None:
                    # First step: use noop control for overlay
                    overlay_control = noop_control()
                else:
                    overlay_control = self.prev_control
                
                obs_with_overlay = self._draw_action_overlay(
                    image_center_rgb.copy(), overlay_control
                )
                self._debug_print_overlay_info(self.steps, success=True)
                overlay_applied = True
            except Exception as e:
                print(f"WARNING: Overlay failed at step {self.steps}: {e}")
                self._debug_print_overlay_info(self.steps, success=False, error=e)
                obs_with_overlay = image_center_rgb.copy()
                overlay_applied = False
        else:
            obs_with_overlay = image_center_rgb.copy()
            overlay_applied = False

        # Record the frame for exported video (with overlay if applied)
        self.frames_to_record.append(obs_with_overlay.copy())
        self.agent_engaged = True

        # Process observation and predict control with confounded image
        final_model_input = self._process_observation(obs_with_overlay)
        control = self._predict_control(final_model_input)

        # For debugging and raw file saving
        stacked_obs_np = np.stack(self.frames_stack, axis=0)
        stacked_obs_torch = torch.from_numpy(stacked_obs_np.copy()).permute(0, 3, 1, 2)
        if self.grayscale and stacked_obs_torch.shape[1] == 3:
            r = stacked_obs_torch[:, 0:1].float()
            gch = stacked_obs_torch[:, 1:2].float()
            b = stacked_obs_torch[:, 2:3].float()
            stacked_obs_torch = 0.299 * r + 0.587 * gch + 0.114 * b
        else:
            stacked_obs_torch = stacked_obs_torch.float()

        # Save raw data if needed
        if self.raw_files != "":
            # Re-compute encoder input for saving
            with torch.no_grad():
                g = None
                if self.gaze_method in ["ViSaRL", "Mask", "AGIL"] or self.dp_method in [
                    "GMD",
                    "IGMD",
                ]:
                    g = self.gaze_predictor(final_model_input)
                    g[g < 0] = 0
                    g[g > 1] = 1

                encoder_input = final_model_input
                if self.gaze_method == "ViSaRL":
                    encoder_input = torch.cat([final_model_input, g], dim=1)
                elif self.gaze_method == "Mask":
                    encoder_input = final_model_input * g

                self.encoder_input_list.append(encoder_input.cpu().numpy())
                z = self.encoder(encoder_input)
                self.z_list.append(z.cpu().numpy())

        v = control_to_vector(control) if hasattr(control, "throttle") else np.zeros(7)

        # Unified debug print
        self._debug_print_step_info(
            step=self.steps,
            input_data=input_data,
            image_center=image_center_rgb,
            obs=obs_with_overlay,
            stacked_obs_np=stacked_obs_np,
            stacked_obs_torch=stacked_obs_torch,
            final_model_input=final_model_input,
            gaze_heatmap=None,  # Will be computed in debug function if needed
            action_vector=v,
            control=control,
        )

        # Store current control for next step's overlay (in confounded mode)
        if self.confounded:
            self.prev_control = control

        # Early cold start: output empty control for first 10 frames to avoid early instability
        if self.steps < 10:
            return noop_control()
        # Stop extremely long episodes to avoid hangs
        elif self.steps > self.fps * 100:
            print(
                f"TIMEOUT: BCAgent exceeded {self.fps * 100} steps (current: {self.steps})"
            )
            print(f"Confounded mode: {self.confounded}")
            print(f"Game time: {self.steps / self.fps:.1f}s")
            raise Exception("BCAgent failed to finish the route")

        return control

    def _draw_action_overlay(self, frame_rgb, control):
        """
        Draw action-based patterns on the frame:
        - Brake red dot (bottom left)
        - Turn arrows (left/right)
        - Straight acceleration arrow (up)
        Note: Input is RGB uint8, uses BGR internally for drawing, then converts back to RGB.
        """
        # Convert to BGR for OpenCV drawing
        img_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        # Thresholds: read from config (use default if none)
        thresholds = (
            (self.confounded_cfg or {}).get("action_processing", {})
            if self.confounded_cfg
            else {}
        )
        brake_light = float(
            ((thresholds.get("braking") or {}).get("light_threshold")) or 0.1
        )
        straight_thr = float(
            ((thresholds.get("steering") or {}).get("straight_threshold")) or 0.05
        )
        throttle_light = float(
            ((thresholds.get("throttle") or {}).get("light_threshold")) or 0.1
        )

        throttle = float(np.clip(getattr(control, "throttle", 0.0), 0.0, 1.0))
        steer = float(np.clip(getattr(control, "steer", 0.0), -1.0, 1.0))
        brake = float(1.0 if getattr(control, "brake", 0.0) else 0.0)

        # Render config
        render = (
            (self.confounded_cfg or {}).get("render", {}) if self.confounded_cfg else {}
        )
        dot_cfg = render.get("dot", {})
        arrow_cfg = render.get("arrow", {})
        anchor = (render.get("anchor") or "bottom_left").lower()

        # Red dot params
        radius = int(dot_cfg.get("radius", 7))
        red_bgr = tuple(int(c) for c in dot_cfg.get("color_bgr", [0, 0, 255]))

        if anchor == "top_mid":
            margin_top = int(dot_cfg.get("margin_top", 10))
            offset_x = int(dot_cfg.get("offset_x", 0))
            extra_down = int(render.get("top_mid_offset_y", 12))
            cx = max(radius, w // 2 + offset_x)
            cy = max(radius + margin_top + extra_down, radius)
        else:
            margin_left = int(dot_cfg.get("margin_left", 10))
            margin_bottom = int(dot_cfg.get("margin_bottom", 10))
            cx = max(radius + margin_left, radius)
            cy = h - max(radius + margin_bottom, radius)

        # Brake red dot (BGR)
        if brake > brake_light and dot_cfg.get("enabled", True):
            cv2.circle(
                img_bgr, (cx, cy), radius, red_bgr, thickness=-1, lineType=cv2.LINE_AA
            )

        # Arrow config
        gap_default = int(arrow_cfg.get("gap_from_dot", 8))
        gap_left = int(arrow_cfg.get("gap_left", gap_default))
        gap_right = int(arrow_cfg.get("gap_right", gap_default))
        y_offset = int(arrow_cfg.get("y_offset", 0))
        base_length = int(arrow_cfg.get("length", 32))
        base_thickness = int(arrow_cfg.get("thickness", 2))
        base_head = int(arrow_cfg.get("head_size", 6))
        color_bgr = tuple(int(c) for c in arrow_cfg.get("color_bgr", [255, 255, 255]))

        # Left/Right turn arrows
        if arrow_cfg.get("enabled", True) and abs(steer) >= straight_thr:
            norm = min(
                1.0,
                max(0.0, (abs(steer) - straight_thr) / max(1e-6, 1.0 - straight_thr)),
            )
            scale = 0.5 + 1.5 * norm
            length = max(6, int(base_length * scale))
            thickness = max(1, int(round(base_thickness * scale)))
            head_size = max(3, int(round(base_head * scale)))
            y = int(np.clip(cy + y_offset, 0, h - 1))
            if steer < 0:
                # Left arrow, pointing left
                end_x = max(0, cx - gap_left)
                start_x = max(0, end_x + length)
                start = (start_x, y)
                end = (end_x, y)
            else:
                # Right arrow, pointing right
                start_x = min(w - 1, cx + gap_right)
                end_x = min(w - 1, start_x + length)
                start = (start_x, y)
                end = (end_x, y)
            cv2.arrowedLine(
                img_bgr,
                start,
                end,
                color_bgr,
                thickness=thickness,
                tipLength=max(0.1, head_size / max(length, 1)),
            )

        # Straight acceleration arrow (pointing up)
        if (
            arrow_cfg.get("enabled", True)
            and abs(steer) < straight_thr
            and throttle > throttle_light
        ):
            norm = min(
                1.0,
                max(0.0, (throttle - throttle_light) / max(1e-6, 1.0 - throttle_light)),
            )
            scale = 0.5 + 1.5 * norm
            length = max(6, int(base_length * scale))
            thickness = max(1, int(round(base_thickness * scale)))
            head_size = max(3, int(round(base_head * scale)))
            start = (cx, max(0, cy - gap_default))
            end = (cx, max(radius, start[1] - length))
            cv2.arrowedLine(
                img_bgr,
                start,
                end,
                color_bgr,
                thickness=thickness,
                tipLength=max(0.1, head_size / max(length, 1)),
            )

        # Convert back to RGB output
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def destroy(self):
        """
        Cleanup
        """

        if self.raw_files != "":
            np.save(self.raw_files + "/z_list.npy", np.array(self.z_list))
            self.z_list = []

            np.save(
                self.raw_files + "/encoder_input_list.npy",
                np.array(self.encoder_input_list),
            )
            self.encoder_input_list = []

        torch.cuda.empty_cache()
        super(BCAgent, self).destroy()
