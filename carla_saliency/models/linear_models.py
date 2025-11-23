import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils.gaze_utils import apply_gmd_dropout


def weight_init(m):
    """Kaiming init for Conv2D and Linear layers (ReLU nonlinearity)."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            # + torch.sum(self._embedding.weight**2, dim=1)
            + torch.sum(self._embedding.weight.t() ** 2, dim=0, keepdim=True)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs, reduction="none").mean(
            (1, 2, 3)
        )
        q_latent_loss = F.mse_loss(quantized, inputs.detach(), reduction="none").mean(
            (1, 2, 3)
        )
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        encoding_indices = encoding_indices.view(input_shape[0], -1)
        return quantized, loss, perplexity, encodings, encoding_indices, distances


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(
        self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
    ):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [
                Residual(in_channels, num_hiddens, num_residual_hiddens)
                for _ in range(self._num_residual_layers)
            ]
        )

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(
        self,
        input_channels,
        embedding_dim,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
    ):
        super(Encoder, self).__init__()
        self._input_channels = input_channels
        self._embedding_dim = embedding_dim
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        ## 90 x 160
        self._conv_1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=num_hiddens // 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        ## 45 x 80
        self._conv_2 = nn.Conv2d(
            in_channels=num_hiddens // 4,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        ## 22 x 40
        self._conv_3 = nn.Conv2d(
            in_channels=num_hiddens // 2,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        ## 20 x 38
        self._conv_4 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        ## 20 x 38
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )
        ## 20 x 38
        self._conv_5 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=(5, 5),
            stride=1,
            padding=5 // 2,
        )

        ## 20 x 38
        self._conv_6 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=embedding_dim,
            kernel_size=(5, 5),
            stride=1,
            padding=5 // 2,
        )
        self.apply(weight_init)

    def forward(self, inputs, dropout_mask=None):
        x = self._conv_1(inputs)
        x = F.relu(x)
        if not (dropout_mask is None):
            x = apply_gmd_dropout(x, dropout_mask, test_mode=not self.training)
        x = self._conv_2(x)
        x = F.relu(x)
        if not (dropout_mask is None):
            x = apply_gmd_dropout(x, dropout_mask, test_mode=not self.training)
        x = self._conv_3(x)
        x = F.relu(x)

        x = self._conv_4(x)
        x = self._residual_stack(x)
        x = self._conv_5(x)
        x = F.relu(x)
        return self._conv_6(x)


class BCActor(nn.Module):
    """Behavior cloning actor head that flattens encoder features and predicts actions."""

    def __init__(self, encoder_output_dim: int, latent_dim: int, action_dim: int):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.fc_in = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_hidden = nn.Linear(latent_dim, latent_dim)
        self.activation = nn.ReLU()
        self.fc_out = nn.Linear(latent_dim, action_dim)

        self.apply(weight_init)

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        latent = self.fc_in(self.flatten(x))
        hidden = self.activation(self.fc_hidden(latent))
        actions = self.fc_out(hidden)

        if return_latent:
            return actions, latent
        return actions


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels,
        embedding_dim,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
    ):
        super(Decoder, self).__init__()
        self._out_channles = out_channels
        self._embedding_dim = embedding_dim
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        ## 20 x 38
        self._conv_1 = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        ## 20 x 38
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )
        ## 22 x 40
        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
        )
        ## 45 x 81
        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=(1, 0),
        )
        ## 90 x 162
        self._conv_trans_3 = nn.ConvTranspose2d(
            in_channels=num_hiddens // 2,
            out_channels=num_hiddens // 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        ## 180 x 324
        self._conv_trans_4 = nn.ConvTranspose2d(
            in_channels=num_hiddens // 4,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.apply(weight_init)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)
        x = self._conv_trans_2(x)
        x = F.relu(x)
        x = self._conv_trans_3(x)
        x = F.relu(x)
        x = self._conv_trans_4(x)
        return x


class VQVAEModel(nn.Module):
    def __init__(self, encoder, decoder, quantizer):
        super(VQVAEModel, self).__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._quantizer = quantizer

    def forward(self, x, encode_only=False):
        z = self._encoder(x)
        quantized, vq_loss, _, _, encoding_indices, _ = self._quantizer(z)
        if encode_only:
            x_recon = None
        else:
            x_recon = self._decoder(quantized)
        return z, x_recon, vq_loss, quantized, encoding_indices


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


class PreActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, z_dim, out_dim):
        super().__init__()
        self.trunk = mlp(z_dim, None, out_dim, 0)
        self.apply(weight_init)

    def forward(self, h):
        h = torch.flatten(h, start_dim=1)
        logits = self.trunk(h)
        return logits


class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, z_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()
        self.trunk = mlp(z_dim, hidden_dim, action_dim, hidden_depth)
        self.apply(weight_init)

    def forward(self, h):
        logits = self.trunk(h)
        return logits


class Projector(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, z_dim, out_dim, hidden_dim, hidden_depth):
        super().__init__()
        self.trunk = mlp(z_dim, hidden_dim, out_dim, hidden_depth)
        self.apply(weight_init)

    def forward(self, h):
        outputs = self.trunk(h)
        return outputs


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, encode_only=False):
        z = self.encoder(x)
        if encode_only:
            return z
        x_recon = self.decoder(z)
        return x_recon


if __name__ == "__main__":
    encoder = Encoder(6, 64, 128, 2, 32)
    decoder = Decoder(6, 64, 128, 2, 32)
    quantizer = VectorQuantizer(64, 512, 0.25)
    x = torch.randn(12, 6, 180, 320)

    assert decoder(encoder(x)).shape == x.shape

    vqvae = VQVAEModel(encoder, decoder, quantizer)

    z, x_recon, vq_loss, quantized, encoding_indices = vqvae(x)
    print(
        z.shape, x_recon.shape, vq_loss.shape, quantized.shape, encoding_indices.shape
    )
