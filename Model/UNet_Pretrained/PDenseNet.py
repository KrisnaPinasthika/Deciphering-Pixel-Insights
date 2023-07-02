import torch
import torch.nn as nn
from torchvision.models import densenet169, DenseNet169_Weights


class DenseNetEncoderBlock(nn.Module):
    """Some Information about EncoderBlock"""

    def __init__(self, freeze=False):
        super(DenseNetEncoderBlock, self).__init__()
        self.densenet = densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)
        
        if freeze:
            for v in self.densenet.parameters():
                v.requires_grad = False
        
        # for v in self.densenet.parameters():
        #     print(v.requires_grad)

    def forward(self, x):
        features = [x]
        # i = 1
        for k, v in self.densenet.features._modules.items():
            features.append(v(features[-1]))
            
        return features

class DecoderBLock(nn.Module):
    """Some Information about DecoderBLock"""

    def __init__(self, input_channel, concatenated_channel, output_channel, kernel_size):
        super(DecoderBLock, self).__init__()
        self.transconv = nn.ConvTranspose2d(
                            in_channels=input_channel,
                            out_channels=input_channel,
                            kernel_size=2,
                            stride=2,
                            padding=0,
                            dilation=1,
                            bias=True,
                        )

        self.conv1 = nn.Conv2d(
                        in_channels=concatenated_channel,
                        out_channels=output_channel,
                        kernel_size=kernel_size,
                        stride=1,
                        padding="same",
                        bias=True,
                        )
        self.conv2 = nn.Conv2d(
                        in_channels=output_channel,
                        out_channels=output_channel,
                        kernel_size=kernel_size,
                        stride=1,
                        padding="same",
                        bias=True,
                    )
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, skip):
        x = self.transconv(x)
        x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)

        return x

class PretrainedUNetDenseNet(nn.Module):
    """Some Information about UNetResNet"""

    def __init__(self, device, freeze=False):
        super(PretrainedUNetDenseNet, self).__init__()
        self.encoder = DenseNetEncoderBlock(freeze=freeze).to(device)
        features = 1664 # channel output dari densenet
        self.decoder = nn.ModuleList([
                            DecoderBLock(input_channel= features//1 ,concatenated_channel=features//1 + 256, output_channel=features//2, kernel_size=3),
                            DecoderBLock(input_channel= features//2,concatenated_channel=features//2 + 128, output_channel=features//4, kernel_size=3),
                            DecoderBLock(input_channel= features//4,concatenated_channel=features//4 + 64,  output_channel=features//8, kernel_size=3),
                            DecoderBLock(input_channel= features//8,concatenated_channel=features//8 + 64,  output_channel=features//16, kernel_size=3),
                        ]).to(device)

        self.head = nn.Sequential(
                        nn.Conv2d(in_channels=features//16, out_channels=features//32, kernel_size=1, stride=1, padding="same"),
                        nn.ConvTranspose2d(
                                    in_channels=features//32,
                                    out_channels=features//64,
                                    kernel_size=2,
                                    stride=2,
                                    padding=0,
                                    dilation=1,
                                    bias=True,),
                        nn.Conv2d(in_channels=features//64, out_channels=1, kernel_size=1, stride=1, padding="same"),
                        nn.Sigmoid()
                    ).to(device)

    def forward(self, x):
        enc = self.encoder(x) 
        block1, block2, block3, block4, block5 = enc[3], enc[4], enc[6], enc[8], enc[12]
        # [1]  : torch.Size([10, 64, 128, 128])
        # [2]  : torch.Size([10, 64, 128, 128])
        #* [3]  : torch.Size([10, 64, 128, 128])    v
        #* [4]  : torch.Size([10, 64, 64, 64])      v
        # [5]  : torch.Size([10, 256, 64, 64])
        #* [6]  : torch.Size([10, 128, 32, 32])     v
        # [7]  : torch.Size([10, 512, 32, 32])
        #* [8]  : torch.Size([10, 256, 16, 16])     v
        # [9]  : torch.Size([10, 1280, 16, 16])
        # [10] : torch.Size([10, 640, 8, 8])
        # [11] : torch.Size([10, 1664, 8, 8])
        #* [12] : torch.Size([10, 1664, 8, 8])      v
        
        u1 = self.decoder[0](block5, block4)
        u2 = self.decoder[1](u1, block3)
        u3 = self.decoder[2](u2, block2)
        u4 = self.decoder[3](u3, block1)

        op = self.head(u4)

        return op


if __name__ == '__main__': 
    # from torchsummary import summary
    # from torchinfo import summaryfrom 
    from prettytable import PrettyTable
    
    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params
    
    model = PretrainedUNetDenseNet(device='cuda')
    # summary(model, input_size=(1, 3, 256, 256))
    # print(model.state_dict())
    count_parameters(model)
