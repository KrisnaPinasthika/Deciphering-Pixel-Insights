import torch
import torch.nn as nn
from torchvision.models import densenet121, densenet161, densenet169, densenet201, DenseNet121_Weights
from torchvision.models import DenseNet121_Weights, DenseNet161_Weights, DenseNet169_Weights, DenseNet201_Weights

class DenseNetEncoderBlock(nn.Module):
    """Some Information about EncoderBlock"""

    def __init__(self, backbone, freeze=False):
        super(DenseNetEncoderBlock, self).__init__()        
        if backbone.lower() == 'densenet121':
            self.densenet = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            
        elif backbone.lower() == 'densenet161':
            self.densenet = densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1)
            
        elif backbone.lower() == 'densenet169':
            self.densenet = densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)
            
        elif backbone.lower() == 'densenet201':
            self.densenet = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
        
        if freeze:
            for v in self.densenet.parameters():
                v.requires_grad = False

    def forward(self, x):
        features = [x]
        encoder = self.densenet.features
        # i = 1
        for layer in encoder:
            features.append(layer(features[-1]))

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

    def forward(self, skip, x):
        x = self.transconv(x)
        x = torch.cat([x, skip], dim=1)

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        
        return x

class AdditiveAttentionGate(nn.Module):
    """Some Information about AttentionGate"""

    def __init__(self, x_channel, g_channel, desired_channel):
        super(AdditiveAttentionGate, self).__init__()
        self.conv_x = nn.Conv2d(in_channels=x_channel,
                                out_channels=desired_channel,
                                kernel_size=1,
                                stride=2)

        self.conv_g = nn.Conv2d(
            in_channels=g_channel,
            out_channels=desired_channel,
            kernel_size=1,
            stride=1,
            padding="same",
        )

        self.psi = nn.Conv2d(
            in_channels=desired_channel,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding="same",
        )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.upsample = nn.ConvTranspose2d(
            in_channels=1,
            out_channels=x_channel,
            kernel_size=2,
            stride=2,
            padding=0,
            dilation=1,
            bias=True,
        )

    def forward(self, x, g):
        # example   x = (128, 64, 64) -> convert to (128, 32, 32) with strided conv
        #           g = (256, 32, 32) -> convert to (128, 32, 32) with 1x1 conv
        wx = self.conv_x(x)
        wg = self.conv_g(g)
        psi = self.psi(self.relu(wx + wg))
        psi = self.sigmoid(psi)
        psi = self.upsample(psi)

        return psi * x

class MonoAGIDenseNet(nn.Module):
    """Some Information about UNetResNet"""

    def __init__(self, device, backbone, freeze=False, act=False):
        super(MonoAGIDenseNet, self).__init__()
        self.encoder = DenseNetEncoderBlock(backbone, freeze).to(device)
        self.resnet_backbone = backbone
        # features = size of last channel
        if backbone.lower() == 'densenet121':
            features = [64, 256, 512, 1024, 1024]
        elif backbone.lower() == 'densenet161':
            features = [96, 384, 768, 2112, 2208]
        elif backbone.lower() == 'densenet169':
            features = [64, 256, 512, 1280, 1664]
        elif backbone.lower() == 'densenet201':
            features = [64, 256, 512, 1792, 1920]
        else:
            print('Check your backbone again ^.^')
            return None
        
        self.attention = nn.ModuleList([
            AdditiveAttentionGate(x_channel=features[-2], g_channel=features[-1], desired_channel=features[-2]),
            AdditiveAttentionGate(x_channel=features[-3], g_channel=features[-2], desired_channel=features[-3]),
            AdditiveAttentionGate(x_channel=features[-4], g_channel=features[-3], desired_channel=features[-4]),
            AdditiveAttentionGate(x_channel=features[-5], g_channel=features[-4], desired_channel=features[-5]),
        ]).to(device)
        
        self.decoder = nn.ModuleList([
            DecoderBLock(input_channel=features[-1],
                            concatenated_channel=features[-1] + features[-2], 
                            output_channel=features[-2], 
                            kernel_size=3),
            DecoderBLock(input_channel= features[-2],
                            concatenated_channel=features[-2] + features[-3], 
                            output_channel=features[-3], 
                            kernel_size=3),
            DecoderBLock(input_channel= features[-3],
                            concatenated_channel=features[-3] + features[-4],  
                            output_channel=features[-4], 
                            kernel_size=3),
            DecoderBLock(input_channel= features[-4],
                            concatenated_channel=features[-4] + features[-5],  
                            output_channel=features[-5], 
                            kernel_size=3),
        ]).to(device)

        if act:
            self.head = nn.Sequential(
                nn.Conv2d(in_channels=features[-5], 
                            out_channels=features[-5]//2, 
                            kernel_size=3, 
                            stride=1, 
                            padding="same"),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(
                            in_channels=features[-5]//2,
                            out_channels=features[-5]//2,
                            kernel_size=2,
                            stride=2,
                            padding=0,
                            dilation=1,
                            bias=True,),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=features[-5]//2, 
                            out_channels=1, 
                            kernel_size=1, 
                            stride=1, 
                            padding="same"),
                nn.Sigmoid()
            ).to(device)
        else: 
            self.head = nn.Sequential(
                nn.Conv2d(in_channels=features[-5], 
                            out_channels=features[-5]//2, 
                            kernel_size=3, 
                            stride=1, 
                            padding="same"),
                nn.ConvTranspose2d(
                            in_channels=features[-5]//2,
                            out_channels=features[-5]//2,
                            kernel_size=2,
                            stride=2,
                            padding=0,
                            dilation=1,
                            bias=True,),
                nn.Conv2d(in_channels=features[-5]//2, 
                            out_channels=1, 
                            kernel_size=1, 
                            stride=1, 
                            padding="same"),
                nn.Sigmoid()
            ).to(device)
            
    def forward(self, x):
        enc = self.encoder(x) 
        block1, block2, block3, block4, block5 = enc[3], enc[5], enc[7], enc[9], enc[12]
                
        ag1 = self.attention[0](block4, block5)
        u1 = self.decoder[0](ag1, block5)
                
        ag2 = self.attention[1](block3, u1)
        u2 = self.decoder[1](ag2, u1)
        
        ag3 = self.attention[2](block2, u2)
        u3 = self.decoder[2](ag3, u2)
        
        ag4 = self.attention[3](block1, u3)
        u4 = self.decoder[3](ag4, u3)
        
        op = self.head(u4)

        return op

if __name__ == '__main__': 
    from prettytable import PrettyTable
    model = MonoAGIDenseNet(device='cuda', backbone='densenet169', act=True)
    # img = torch.randn(size=(5, 3, 256, 256)).to('cuda')
    # print(model(img).shape)
    # print('--'*20)
    
    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        # print(table)
        print(f"Total Trainable Params: {total_params:,}")
        return total_params
    
    count_parameters(model)