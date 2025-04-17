import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.D_0 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
        )
        self.D = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3, 3), stride=1),
        )

    def forward(self, x):
        out_f = self.D_0(x)
        out = self.D(out_f)
        return out, out_f


class Enhance(nn.Module):
    def __init__(self):
        super(Enhance, self).__init__()
        self.E = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.E(x)
        return out


class base_pan(nn.Module):
    def __init__(self):
        super(base_pan, self).__init__()
        self.B = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.B(x)
        return out

class base_ms(nn.Module):
    def __init__(self):
        super(base_ms, self).__init__()
        self.B = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.B(x)
        return out


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.E = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.E(x)
        return out


class DictionaryRepresentationModule(nn.Module):
    def __init__(self):
        super(DictionaryRepresentationModule, self).__init__()
        element_size = 4
        channel = 8
        l_n = 16
        c_n = 16
        self.Dictionary = nn.Parameter(
            torch.FloatTensor(l_n * c_n, 1, element_size * element_size * channel).to(torch.device("cuda:0")),
            requires_grad=True)
        nn.init.uniform_(self.Dictionary, 0, 1)
        self.unflod_win = nn.Unfold(kernel_size=(element_size, element_size), stride=element_size)
        self.flod_pan = nn.Fold(output_size=(64,64), kernel_size=(element_size, element_size), stride=element_size)
        self.flod_ms = nn.Fold(output_size=(16,16), kernel_size=(element_size, element_size), stride=element_size)
        self.CA = nn.MultiheadAttention(embed_dim=element_size * element_size * channel, num_heads=1, dropout=0)
        self.flod_win_1 = nn.Fold(output_size=(l_n * element_size, c_n * element_size),
                                  kernel_size=(element_size, element_size), stride=element_size)

    def forward(self, x):
        size = x.size()
        D = self.Dictionary.repeat(1, size[0], 1) # torch.Size([256, 128, 128])
        x_w = self.unflod_win(x).permute(2, 0, 1) # torch.Size([256, 128, 128])

        q = x_w
        k = D
        v = D
        a = self.CA(q, k, v)[0] #torch.Size([256, 128, 128])
        
        if size[2]==64:
            representation = self.flod_pan(a.permute(1, 2, 0)) # torch.Size([128, 8, 64, 64])
        elif size[2]==16:
            representation = self.flod_ms(a.permute(1, 2, 0))
        visible_D = self.flod_win_1(self.Dictionary.permute(1, 2, 0)) # torch.Size([1, 8, 64, 64])

        return representation, visible_D
    
class FE(nn.Module):
    def __init__(self, in_channels, feature_dim=128):
        super(FE, self).__init__()
        
        # 获取 ResNet50 结构
        resnet = resnet50(pretrained=False)
        
        # 替换第一个卷积层以适应输入通道数
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 提取 ResNet50 的不同阶段
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1  # conv2_x
        self.layer2 = resnet.layer2  # conv3_x
        self.layer3 = resnet.layer3  # conv4_x
        self.layer4 = resnet.layer4  # conv5_x
        
        # 每个 stage 后的投影头
        self.proj_head3 = self._make_projection_head(512, feature_dim)  # 对应 conv3_x
        self.proj_head4 = self._make_projection_head(1024, feature_dim) # 对应 conv4_x
        self.proj_head5 = self._make_projection_head(2048, feature_dim) # 对应 conv5_x
    
    def _make_projection_head(self, in_dim, feature_dim):
        return nn.Sequential(
            nn.Linear(in_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True)
        )
    
    def forward(self, x):
        x = self.stem(x) #[128,64,16,16] [128,64,16,16]
        x = self.layer1(x)  # conv2_x [128,256,16,16] [128,256,16,16]
        feat3 = self.layer2(x)  # conv3_x [128,512,8,8] [128,512,8,8]
        feat4 = self.layer3(feat3)  # conv4_x [128,1024,4,4] [128,1024,4,4]
        feat5_1 = self.layer4(feat4)  # conv5_x [128,2048,2,2] [128,2048,2,2]
        
        feat3 = F.adaptive_avg_pool2d(feat3, (1, 1)) #[128,512,1,1] 
        feat4 = F.adaptive_avg_pool2d(feat4, (1, 1)) #[128,1024,1,1]
        feat5_2 = F.adaptive_avg_pool2d(feat5_1, (1, 1)) #[128,2048,1,1]

        # 展平特征  3:[128,512] 4:[128,1024] 5:[128,2048]
        feat3 = torch.flatten(feat3, start_dim=1)
        feat4 = torch.flatten(feat4, start_dim=1)
        feat5 = torch.flatten(feat5_2, start_dim=1)
        
        # 通过投影头
        out3 = F.normalize(self.proj_head3(feat3), dim=-1) #[128,128]
        out4 = F.normalize(self.proj_head4(feat4), dim=-1) #[128,128]
        out5 = F.normalize(self.proj_head5(feat5), dim=-1) #[128,128]
        
        return out3, out4, out5, feat5_2

# PAN 模型
class PAN_Model(FE):
    def __init__(self, feature_dim=128):
        super(PAN_Model, self).__init__(in_channels=8, feature_dim=feature_dim)

# MS 模型
class MS_Model(FE):
    def __init__(self, feature_dim=128):
        super(MS_Model, self).__init__(in_channels=8, feature_dim=feature_dim)