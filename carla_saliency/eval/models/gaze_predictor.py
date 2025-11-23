# Gaze predictor model
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, input_channels, output_channels=None):
        super(UNet, self).__init__()
        if output_channels is None:
            output_channels = input_channels

        # Encoder (downsampling path)
        self.enc1 = self.conv_block(input_channels, 8)
        self.enc2 = self.conv_block(8, 16)
        self.enc3 = self.conv_block(16, 16)
        self.enc4 = self.conv_block(16, 32)

        # Bottleneck
        self.bottleneck = self.conv_block(32, 32)

        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(64, 32)

        self.upconv3 = nn.ConvTranspose2d(
            32, 16, kernel_size=2, stride=2, output_padding=(1, 0)
        )
        self.dec3 = self.conv_block(32, 16)

        self.upconv2 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(32, 16)

        self.upconv1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(16, 8)

        # Final output layer (1x1 convolution with fixed output channels)
        self.final_conv = nn.Conv2d(8, output_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """Block of two convolutional layers with batch norm and ReLU."""
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        return block

    def forward(self, x):
        # Encoder (downsampling path with max-pooling)
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoder (upsampling path with concatenation of encoder outputs)
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        # Final 1x1 convolution to get the desired output shape
        out = self.final_conv(dec1)
        return out


class Model(nn.Module):
    def __init__(
        self, in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=1
    ):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        return x


if __name__ == "__main__":
    input_channels = 4 * 3  # 4 frames of RGB images
    model = UNet(input_channels, output_channels=4).to("cuda")
    input_tensor = torch.randn(128, input_channels, 180, 320).to("cuda")
    output_tensor = model(input_tensor)
    print(output_tensor.shape, input_tensor.shape)
    assert output_tensor.shape[-2:] == input_tensor.shape[-2:]
