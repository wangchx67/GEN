import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import argparse

class GEN(nn.Module):
    def __init__(self,opt=None,in_ch=3,intensity=768,out_ch=None):

        super(GEN, self).__init__()

        self.intensity=intensity
        self.out_ch=in_ch if out_ch is None else out_ch
        self.low_img_size = opt.low_img_size
        self.gen=backbone_gen(in_ch=in_ch,intensity=intensity,opt=opt)

        self.apply(weight_init)


    def forward(self, x):

        intensity=int(self.intensity/self.out_ch)-1

        x_=F.interpolate(x,size=(self.low_img_size,self.low_img_size))

        coeffi=self.gen(x_)
        coeffi = coeffi.squeeze(-1).squeeze(-1)
        out = intensity_transform(x, coeffi, intensity=intensity)

        return out

class backbone_gen(nn.Module):
    def __init__(self,in_ch=3,intensity=768,opt=None):
        super(backbone_gen, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 16, 5, 1, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
        )
        self.conv2 = Inverted_Residual_Block(16,24,5,2,6)
        self.conv3 = Inverted_Residual_Block(24,40,5,2,6)
        self.conv4 = Inverted_Residual_Block(40,80,5,2,6)
        self.conv5 = Inverted_Residual_Block(80,112,5,2,6)
        self.out = nn.Sequential(
            nn.Conv2d(112, intensity, 1, 1, bias=False),
            nn.BatchNorm2d(intensity),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(intensity, intensity, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        out=self.out(x5)
        return out


class Inverted_Residual_Block(nn.Module):
    def __init__(self, in_channel, out_channel,kernel_size=3,stride=1,expand_ratio=1,se_ratio=0.25):
        super(Inverted_Residual_Block, self).__init__()

        channel=in_channel*expand_ratio
        self.if_se=False

        self.in_cnn = nn.Sequential(
            nn.Conv2d(in_channel, channel, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(channel),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=(2, 2),groups=channel, bias=False),
            nn.BatchNorm2d(channel),
            nn.SiLU(inplace=True),
        )
        self.se=nn.Identity()
        if 0 < se_ratio <= 1:
            self.if_se = True
            channel_se = max(1, int(in_channel * se_ratio))
            self.se=nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel,channel_se, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(channel_se,channel, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.Sigmoid()
            )
        self.out_cnn=nn.Sequential(
            nn.Conv2d(channel,out_channel, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        x_in=self.in_cnn(x)
        if self.if_se:
            se=self.se(x_in)
            x_in=x_in*se
        x_out=self.out_cnn(x_in)
        return x_out


def intensity_transform(img,coeffi,intensity=256):# img: b,c,h,w  coeffi: b,768?

    B,C,H,W=img.shape
    img_indice=(img*(intensity-1)).long()
    out=torch.empty(img.shape).to(img.device)
    for b in range(B):
        for c in range(C):
            indice = img_indice[b, c, :, :] + (c * intensity)
            out[b, c, :, :] = coeffi[b, indice]
    return out

def weight_init(module):
    for n, m in module.named_children():
        # print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.Tanh):
            pass
        elif isinstance(m, nn.LeakyReLU):
            pass
        else:
            weight_init(m)

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model params
    parser.add_argument('--gpu_id', type=int, default=3)
    parser.add_argument('--CNN_type', type=str, default='GEN')
    parser.add_argument('--intensity', type=int, default=768)
    parser.add_argument('--patch_size', type=int, default=1)
    config = parser.parse_args()

    model=GEN(config)
    a=torch.ones(1,3,256,256)
    OUT=model(a)
    print('end')
    print(count_param(model))

