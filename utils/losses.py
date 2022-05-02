import torch.nn as nn
from torchvision.models.vgg import vgg16
import torch.nn.functional as F


class perception_loss(nn.Module):
    def __init__(self,device=None):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features.to(device)
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out1 = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)

        h = self.to_relu_1_2(y)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out2 = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)

        loss=F.l1_loss(out1[0],out2[0])\
             +F.l1_loss(out1[1],out2[1])\
             +F.l1_loss(out1[2],out2[2])
        return loss

