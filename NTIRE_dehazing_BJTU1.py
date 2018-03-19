import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable


class Dense_block(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(Dense_block, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class Upsample_Block(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(Upsample_Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)

class BJTU_dehaze1(nn.Module):
    def __init__(self):
        super(BJTU_dehaze1, self).__init__()

        haze_class = models.densenet121(pretrained=True)

        self.conv0=haze_class.features.conv0
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0

        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        self.dense_block4=Dense_block(512,256)
        self.trans_block4=Upsample_Block(768,128)

        self.dense_block5=Dense_block(384,256)
        self.trans_block5=Upsample_Block(640,128)

        self.dense_block6=Dense_block(256,128)
        self.trans_block6=Upsample_Block(384,64)

        self.dense_block7=Dense_block(64,64)
        self.trans_block7=Upsample_Block(128,32)

        self.dense_block8=Dense_block(32,32)
        self.trans_block8=Upsample_Block(64,16)

        self.conv_refin=nn.Conv2d(19,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):

        x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))

        x1=self.dense_block1(x0)

        x1=self.trans_block1(x1)

        x2=self.trans_block2(self.dense_block2(x1))

        x3=self.trans_block3(self.dense_block3(x2))


        x4=self.trans_block4(self.dense_block4(x3))

        x42=torch.cat([x4,x2],1)
        x5=self.trans_block5(self.dense_block5(x42))

        x52=torch.cat([x5,x1],1)

        x6=self.trans_block6(self.dense_block6(x52))

        x7=self.trans_block7(self.dense_block7(x6))

        x8=self.trans_block8(self.dense_block8(x7))

        x8=torch.cat([x8,x],1)


        x9=self.relu(self.conv_refin(x8))

        shape_out = x9.data.size()
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)

        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        dehaze = self.tanh(self.refine3(dehaze))

        return dehaze

class BJTU_dehaze11(nn.Module):
    def __init__(self):
        super(BJTU_dehaze11, self).__init__()

        haze_class = models.densenet121(pretrained=True)

        self.juanji0=haze_class.features.conv0
        self.guiyi0=haze_class.features.norm0
        self.nonlinear0=haze_class.features.relu0
        self.chihua0=haze_class.features.pool0

        self.dnblock1=haze_class.features.denseblock1
        self.tnblock1=haze_class.features.transition1

        self.dnblock2=haze_class.features.denseblock2
        self.tnblock2=haze_class.features.transition2

        self.dnblock3=haze_class.features.denseblock3
        self.tnblock3=haze_class.features.transition3

        self.dnblock4=Dense_block(512,256)
        self.tnblock4=Upsample_Block(768,128)

        self.dnblock5=Dense_block(384,256)
        self.tnblock5=Upsample_Block(640,128)

        self.dnblock6=Dense_block(256,128)
        self.tnblock6=Upsample_Block(384,64)

        self.dnblock7=Dense_block(64,64)
        self.tnblock7=Upsample_Block(128,32)

        self.dnblock8=Dense_block(32,32)
        self.tnblock8=Upsample_Block(64,16)

        self.network_def=nn.Conv2d(19,20,3,1,1)
        self.Tanh=nn.Tanh()


        self.juanji1 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.juanji2 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.juanji3 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.juanji4 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.juanji_def = nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)

        self.fangda = F.upsample_nearest

        self.ReLU=nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):

        x0=self.chihua0(self.nonlinear0(self.guiyi0(self.juanji0(x))))

        x1=self.dnblock1(x0)

        x1=self.tnblock1(x1)

        x2=self.tnblock2(self.dnblock2(x1))

        x3=self.tnblock3(self.dnblock3(x2))


        x4=self.tnblock4(self.dnblock4(x3))

        x42=torch.cat([x4,x2],1)
        x5=self.tnblock5(self.dnblock5(x42))

        x52=torch.cat([x5,x1],1)

        x6=self.tnblock6(self.dnblock6(x52))

        x7=self.tnblock7(self.dnblock7(x6))

        x8=self.tnblock8(self.dnblock8(x7))

        x8=torch.cat([x8,x],1)


        x9=self.ReLU(self.network_def(x8))

        shape_out = x9.data.size()
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)

        x1010 = self.fangda(self.ReLU(self.juanji1(x101)), size=shape_out)
        x1020 = self.fangda(self.ReLU(self.juanji2(x102)), size=shape_out)
        x1030 = self.fangda(self.ReLU(self.juanji3(x103)), size=shape_out)
        x1040 = self.fangda(self.ReLU(self.juanji4(x104)), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        dehaze = self.Tanh(self.juanji_def(dehaze))

        return dehaze



