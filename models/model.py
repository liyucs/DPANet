
import torch
import math
import functools
import torch.nn as nn
from utils.padding import Conv2d
import torch.nn.functional as F
import utils.arch_util as arch_util
from utils.modules.deform_conv import ModulatedDeformConv
from utils.utils_flow import bilinear_sampler, coords_grid
from correlation import correlation

#########UPP funcs#####################################################

    
class DOWN(nn.Module):
    def __init__(self):
        super(DOWN, self).__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x2 = self.downsample(x)
        x3 = self.downsample(x2)
        x4 = self.downsample(x3)
        x5 = self.downsample(x4)
        return x,x2,x3,x4,x5

class CONV_BLOCK(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(CONV_BLOCK,self).__init__()
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=out_ch)
        self.RB_1 = basic_block()
        self.RB_2 = basic_block()
        self.conv_block = nn.Sequential(
            Conv2d(in_ch, out_ch, 3),
            nn.ReLU(inplace=True))
    def forward(self,x):
        return self.RB_2(self.RB_1(self.conv_block(x)))

class Offset_Gene(nn.Module):
    def __init__(self,in_ch):
        super(Offset_Gene,self).__init__()
        self.offset_conv = nn.Conv2d(in_ch*2, 27, 3, padding = 1)
        
    def forward(self,xl,xr,cost_volume=None):
        out = self.offset_conv(torch.cat([xl,xr,cost_volume], dim=1))
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return offset, mask
        
class Encoder_Block(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Encoder_Block, self).__init__()
        self.block = CONV_BLOCK(in_ch,out_ch)
    def forward(self,x):    
        x = self.block(x)
        return x
        
class Encoder(nn.Module):
    def __init__(self,in_ch,out_ch,if_down=True):
        super(Encoder,self).__init__()
        self.if_down = if_down
        if self.if_down:
            self.max = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = Encoder_Block(in_ch, out_ch)
        self.offset_l = Offset_Gene(out_ch//2*3)
        self.offset_r = Offset_Gene(out_ch//2*3)
        self.flow_conv = nn.Sequential(
            Conv2d(81, out_ch, 1),
            nn.ReLU(inplace=True),
            Conv2d(out_ch, out_ch, 1),
            nn.ReLU(inplace=True))
            
        self.deform = ModulatedDeformConv(out_ch, out_ch, 3, padding=1) 
    def forward(self,xl,xr,cost_volume=None):
        if self.if_down:
            xl = self.max(xl)
            xr = self.max(xr)
            cost_volume = self.max(cost_volume)
        xl = self.block(xl)
        xr = self.block(xr)

        cost_volume = self.flow_conv(cost_volume)
        off_l, m_l = self.offset_l(xl,xr,cost_volume)
        off_r, m_r = self.offset_r(xl,xr,cost_volume)
        xl = self.deform(xl,off_l,m_l)
        xr = self.deform(xr,off_r,m_r)
        return xl, xr
        
class Deform_Encoder(nn.Module):
    def __init__(self):
        super(Deform_Encoder,self).__init__()
        self.down = DOWN()
        self.encoder1 = Encoder(64, 64,if_down=False)
        self.encoder2 = Encoder(64, 128)
        self.encoder3 = Encoder(128, 256)
        self.encoder4 = Encoder(256, 512) 
        
    def forward(self,xl,xr,cv0):
        xl1,xr1 = self.encoder1(xl,xr,cost_volume=cv0)
        cv1 = F.leaky_relu(input=correlation.FunctionCorrelation(tenFirst=xl1, tenSecond=xr1),
                                 negative_slope=0.1, inplace=False)
        xl2,xr2 = self.encoder2(xl1,xr1,cost_volume=cv1)
        cv2 = F.leaky_relu(input=correlation.FunctionCorrelation(tenFirst=xl2, tenSecond=xr2),
                         negative_slope=0.1, inplace=False)
        xl3,xr3 = self.encoder3(xl2,xr2,cost_volume=cv2)
        cv3 = F.leaky_relu(input=correlation.FunctionCorrelation(tenFirst=xl3, tenSecond=xr3),
                 negative_slope=0.1, inplace=False)
        
        xl4,xr4 = self.encoder4(xl3,xr3,cost_volume=cv3)
        return xl1,xr1,xl2,xr2,xl3,xr3,xl4,xr4
    
class UP_SAMPLE(nn.Module):
    def __init__(self,in_ch):
        super(UP_SAMPLE,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2d(in_ch,in_ch//2,2))
    def forward(self,x):
        return self.up(x)
    
    
class UP(nn.Module):
    def __init__(self,in_ch):
        super(UP,self).__init__()
        self.upsample = UP_SAMPLE(in_ch)
        
        self.offset_l = nn.Conv2d(in_ch//2*3, 27, 3, padding = 1)
        self.offset_r = nn.Conv2d(in_ch//2*3, 27, 3, padding = 1)
        self.deform_l = ModulatedDeformConv(in_ch//2, in_ch//2, 3, padding=1) 
        self.deform_r = ModulatedDeformConv(in_ch//2, in_ch//2, 3, padding=1) 
        
        self.cv = CONV_BLOCK(in_ch//2*3, in_ch//2)
        
    def forward(self,xd,xl,xr):
        xd = self.upsample(xd)

        diffY = xl.size()[2] - xd.size()[2]
        diffX = xl.size()[3] - xd.size()[3]

        xd = F.pad(xd, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        out_l = self.offset_l(torch.cat([xl,xr,xd], dim=1))
        o1_l, o2_l, mask_l = torch.chunk(out_l, 3, dim=1)
        offset_l = torch.cat((o1_l, o2_l), dim=1)
        mask_l = torch.sigmoid(mask_l)
        xl = self.deform_l(xl,offset_l,mask_l)
        
        out_r = self.offset_r(torch.cat([xl,xr,xd], dim=1))
        o1_r, o2_r, mask_r = torch.chunk(out_r, 3, dim=1)
        offset_r = torch.cat((o1_r, o2_r), dim=1)
        mask_r = torch.sigmoid(mask_r)
        xr = self.deform_r(xr,offset_r,mask_r)
          
        x = torch.cat([xl,xr,xd], dim=1)
        x = self.cv(x)
        return x
    
    
class Predeblur_ResNet_Pyramid(nn.Module):
    def __init__(self, nf=64):
        super(Predeblur_ResNet_Pyramid, self).__init__()

        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.RB_L1_1 = basic_block()
        self.RB_L1_2 = basic_block()
        self.RB_L1_3 = basic_block()
        self.RB_L1_4 = basic_block()
        self.RB_L1_5 = basic_block()
        self.RB_L2_1 = basic_block()
        self.RB_L2_2 = basic_block()
        self.RB_L3_1 = basic_block()
        self.deblur_L2_conv = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.deblur_L3_conv = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):

        L1_fea = self.lrelu(self.conv_first(x))
        L2_fea = self.lrelu(self.deblur_L2_conv(L1_fea))
        L3_fea = self.lrelu(self.deblur_L3_conv(L2_fea))
        L3_fea = F.interpolate(self.RB_L3_1(L3_fea), scale_factor=2, mode='bilinear',
                               align_corners=False)
        L2_fea = self.RB_L2_1(L2_fea) + L3_fea
        L2_fea = F.interpolate(self.RB_L2_2(L2_fea), scale_factor=2, mode='bilinear',
                               align_corners=False)
        L1_fea = self.RB_L1_2(self.RB_L1_1(L1_fea)) + L2_fea
        out = self.RB_L1_5(self.RB_L1_4(self.RB_L1_3(L1_fea)))
        return out
    
class VGG_Deblur(nn.Module):
    def __init__(self):
        super(VGG_Deblur, self).__init__()        
        self.predeblur = Predeblur_ResNet_Pyramid()
        
        self.encoder = Deform_Encoder()
        self.encoder_5 = Encoder_Block(512, 512)
        
        self.up4 = UP(1024)
        self.up3 = UP(512)
        self.up2 = UP(256)
        self.up1 = UP(128)
        
        self.out = nn.Sequential(
            Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            Conv2d(64, 3, 3))

        self.__init_weight()
        self.down = DOWN()
 
        
    def forward(self, x_l, x_r):
        x_l_deblur = self.predeblur(x_l)
        x_r_deblur = self.predeblur(x_r)
        tenVolume0 = F.leaky_relu(input=correlation.FunctionCorrelation(tenFirst=x_l_deblur, tenSecond=x_r_deblur),
                                 negative_slope=0.1, inplace=False)
        
        xl1,xr1,xl2,xr2,xl3,xr3,xl4,xr4 = self.encoder(x_l_deblur,x_r_deblur,tenVolume0)
        xl5 = self.encoder_5(xl4)
        xr5 = self.encoder_5(xr4)
        
        xd = torch.cat([xl5,xr5], dim=1)
        y4 = self.up4(xd,xl4,xr4)
        y3 = self.up3(y4,xl3,xr3)
        y2 = self.up2(y3,xl2,xr2)
        y1 = self.up1(y2,xl1,xr1)
        out = self.out(y1)
        
        return torch.sigmoid(out)

    
    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight,mode = 'fan_out',nonlinearity='relu')
            
    
  
                
                
                
                
                
                
                
                
                
                
                
                
                