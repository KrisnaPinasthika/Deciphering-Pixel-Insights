import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l

class EncoderBlock(nn.Module):
    """Some Information about EncoderBlock"""

    def __init__(self, backbone_name, freeze=False):
        super(EncoderBlock, self).__init__()        
        if backbone_name.lower() == 'efficient_v2_s':
            self.backbone = efficientnet_v2_s(pretrained=True)
            
        elif backbone_name.lower() == 'efficient_v2_m':
            self.backbone = efficientnet_v2_m(pretrained=True)
            
        elif backbone_name.lower() == 'efficient_v2_l':
            self.backbone = efficientnet_v2_l(pretrained=True)
            
        if freeze:
            for v in self.backbone.parameters():
                v.requires_grad = False

    def forward(self, x):
        features = [x]
        encoder = self.backbone.features
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

        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)
        
        return x

class UNetEfficientV2(nn.Module):
    """Some Information about UNetResNet"""

    def __init__(self, device, backbone_name, freeze=False):
        super(UNetEfficientV2, self).__init__()
        self.encoder = EncoderBlock(backbone_name, freeze).to(device)
        self.backbone_name = backbone_name
        # features = size of last channel
        if backbone_name.lower() == 'efficient_v2_s':
            features = [24, 48, 64, 160, 256]
        elif backbone_name.lower() == 'efficient_v2_m':
            features = [24, 48, 80, 176, 304]
        elif backbone_name.lower() == 'efficient_v2_l':
            features = [32, 64, 96, 224, 384]
        else:
            print('Check your backbone again ^.^')
            return None
                
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
        block1, block2, block3, block4, block5 = enc[2], enc[3], enc[4], enc[6], enc[7]
                
        u1 = self.decoder[0](block4, block5)
        u2 = self.decoder[1](block3, u1)
        u3 = self.decoder[2](block2, u2)
        u4 = self.decoder[3](block1, u3)
        
        op = self.head(u4)

        return op

if __name__ == '__main__': 
    from torchsummary import summary
    
    model = UNetEfficientV2(device='cuda', backbone_name='efficient_v2_l').to('cuda')
    summary(model, (3, 192, 256))
    
    # img = torch.randn((10, 3, 192, 256)).to('cuda')
    # print(model(img).shape)