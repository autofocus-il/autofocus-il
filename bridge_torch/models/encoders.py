import torch
import torch.nn as nn
import torchvision.models as tvm


class ResNetV1Bridge(nn.Module):
    def __init__(
        self,
        arch: str = "resnet34",
    ):
        super().__init__()

        base = {
            "resnet18": tvm.resnet18(weights=None),
            "resnet34": tvm.resnet34(weights=None),
            "resnet50": tvm.resnet50(weights=None),
            "resnet101": tvm.resnet101(weights=None),
            "resnet152": tvm.resnet152(weights=None),
        }[arch]
        # keep stem and layers; drop avgpool/fc
        # print(f"base: {base}")
        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.avgpool = base.avgpool
        # dummy_input = torch.zeros(1, 3, 256, 256)
        # dummy_output = self.stem(dummy_input)
        # print(f"dummy_output: {dummy_output.shape}")
        
        self._in_ch = None
        self.num_output_features = None
        
    def adapt_to_input_channels(self, x: torch.Tensor):
        """Adapt conv1 if input channels do not match the incoming tensor.

        This modifies the first convolution to accept arbitrary input channels
        by replicating the average across original RGB weights.
        """
        in_ch = int(x.shape[1])
        conv1: nn.Conv2d = self.stem[0]

        if isinstance(conv1, nn.Conv2d) and conv1.in_channels != in_ch:
            new = nn.Conv2d(
                in_ch,
                conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=False,
            )
            with torch.no_grad():
                w = conv1.weight
                mean_w = w.mean(dim=1, keepdim=True)  # average across channels
                new_w = mean_w.repeat(1, in_ch, 1, 1)
                new.weight.copy_(new_w)
            self.stem[0] = new.to(conv1.weight.device)
            self._in_ch = in_ch
            
        self.output_features = self.forward(x).flatten(1).shape[1]
        return self.output_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        self.z_map = z
        x = self.avgpool(z)
        return x