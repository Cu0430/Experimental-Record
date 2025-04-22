import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.D_0 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(64),
            
        )
        self.D = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), stride=1),
        )

    def forward(self, x):
        out_f = self.D_0(x)
        out = self.D(out_f)
        return out, out_f


class DictionaryRepresentationModule_p(nn.Module):
    def __init__(self):
        super(DictionaryRepresentationModule_p, self).__init__()
        element_size = 4
        channel = 64
        l_n, c_n = 4,4
        self.Dictionary = nn.Parameter(
            torch.FloatTensor(l_n * c_n, 1, element_size * element_size * channel).to(torch.device("cuda:0")),
            requires_grad=True)
        nn.init.xavier_uniform_(self.Dictionary)
        self.unflod_win = nn.Unfold(kernel_size=(element_size, element_size), stride=element_size)
        self.flod_pan = nn.Fold(output_size=(16,16), kernel_size=(element_size, element_size), stride=element_size)
        self.flod_ms = nn.Fold(output_size=(4,4), kernel_size=(element_size, element_size), stride=element_size)
        self.CA = nn.MultiheadAttention(embed_dim=element_size * element_size * channel, num_heads=1, dropout=0.1)
        self.flod_win_1 = nn.Fold(output_size=(l_n * element_size, c_n * element_size),
                                  kernel_size=(element_size, element_size), stride=element_size)
        self.norm = nn.LayerNorm(element_size * element_size * channel)
        self.post_proj = nn.Sequential(
                    nn.Linear(element_size * element_size * channel, element_size * element_size * channel),
                    nn.ReLU(inplace=True),nn.Dropout(p=0.1),
                    nn.Linear(element_size * element_size * channel, element_size * element_size * channel)
                        )
        
    def forward(self, x):
        size = x.size()
        D = self.Dictionary.repeat(1, size[0], 1) # torch.Size([256, 128, 128])
        x_w = self.unflod_win(x).permute(2, 0, 1) # torch.Size([256, 128, 128])

        q = x_w
        k = D
        v = D
        a = self.CA(q, k, v)[0] #torch.Size([256, 128, 128])
        #a = self.norm(a)
        #a = self.post_proj(a)
        representation = self.flod_pan(a.permute(1, 2, 0)) # torch.Size([128, 8, 64, 64])
        representation = representation + x
        visible_D = self.flod_win_1(self.Dictionary.permute(1, 2, 0)) # torch.Size([1, 8, 64, 64])

        return representation, visible_D

class DictionaryRepresentationModule_m(nn.Module):
    def __init__(self):
        super(DictionaryRepresentationModule_m, self).__init__()
        element_size = 2
        channel = 64
        l_n, c_n = 2,2
        self.Dictionary = nn.Parameter(
            torch.FloatTensor(l_n * c_n, 1, element_size * element_size * channel).to(torch.device("cuda:0")),
            requires_grad=True)
        nn.init.xavier_uniform_(self.Dictionary)
        self.unflod_win = nn.Unfold(kernel_size=(element_size, element_size), stride=element_size)
        self.flod_ms = nn.Fold(output_size=(4,4), kernel_size=(element_size, element_size), stride=element_size)
        self.CA = nn.MultiheadAttention(embed_dim=element_size * element_size * channel, num_heads=1, dropout=0.1)
        self.flod_win_1 = nn.Fold(output_size=(l_n * element_size, c_n * element_size),
                                  kernel_size=(element_size, element_size), stride=element_size)
        self.norm = nn.LayerNorm(element_size * element_size * channel)
        self.post_proj = nn.Sequential(
                    nn.Linear(element_size * element_size * channel, element_size * element_size * channel),
                    nn.ReLU(inplace=True),nn.Dropout(p=0.1),
                    nn.Linear(element_size * element_size * channel, element_size * element_size * channel)
                        )
        
    def forward(self, x):
        size = x.size()
        D = self.Dictionary.repeat(1, size[0], 1) # torch.Size([256, 128, 128])
        x_w = self.unflod_win(x).permute(2, 0, 1) # torch.Size([256, 128, 128])

        q = x_w
        k = D
        v = D
        a = self.CA(q, k, v)[0] #torch.Size([256, 128, 128])
        #a = self.norm(a)
        #a = self.post_proj(a)
        representation = self.flod_ms(a.permute(1, 2, 0)) # torch.Size([128, 8, 64, 64])
        visible_D = self.flod_win_1(self.Dictionary.permute(1, 2, 0)) # torch.Size([1, 8, 64, 64])
        representation = representation + x
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
    
    def forward_stem(self,x):
        x_1 = self.stem(x) #[128,64,4,4] [128,64,16,16]
          
        return x_1

    def forward(self, x):
        x=self.layer1(x)# [128,256,4,4] [128,256,16,16]
        feat3_1 = self.layer2(x)  # conv3_x [128,512,2,2] [128,512,8,8]
        feat4_1 = self.layer3(feat3_1)  # conv4_x [128,1024,1,1] [128,1024,4,4]
        feat5_1 = self.layer4(feat4_1)  # conv5_x [128,2048,1,1] [128,2048,2,2]
        
        feat3_2 = F.adaptive_avg_pool2d(feat3_1, (1, 1)) #[128,512,1,1] 
        feat4_2 = F.adaptive_avg_pool2d(feat4_1, (1, 1)) #[128,1024,1,1]
        feat5_2 = F.adaptive_avg_pool2d(feat5_1, (1, 1)) #[128,2048,1,1]

        # 展平特征  3:[128,512] 4:[128,1024] 5:[128,2048]
        feat3 = torch.flatten(feat3_2, start_dim=1)
        feat4 = torch.flatten(feat4_2, start_dim=1)
        feat5 = torch.flatten(feat5_2, start_dim=1)
        
        # 通过投影头
        out3 = F.normalize(self.proj_head3(feat3), dim=-1) #[128,128]
        out4 = F.normalize(self.proj_head4(feat4), dim=-1) #[128,128]
        out5 = F.normalize(self.proj_head5(feat5), dim=-1) #[128,128]
        
        return out3, out4, out5, feat3_1,feat4_1,feat5_1

# PAN 模型
class PAN_Model(FE):
    def __init__(self, feature_dim=128):
        super(PAN_Model, self).__init__(in_channels=1, feature_dim=feature_dim)

# MS 模型
class MS_Model(FE):
    def __init__(self, feature_dim=128):
        super(MS_Model, self).__init__(in_channels=4, feature_dim=feature_dim)


class ModalityAwareFPN(nn.Module):
    def __init__(self, out_channels=256):
        super(ModalityAwareFPN, self).__init__()
        self.initialized = False
        self.out_channels = out_channels

    def _build_layers(self, ms_feats, pan_feats):
        self.attn_convs = nn.ModuleList()
        self.lateral_convs = nn.ModuleList()
        self.smooth_convs = nn.ModuleList()

        for ms_feat, pan_feat in zip(ms_feats, pan_feats):
            in_ch = ms_feat.shape[1]
            # 模态注意力融合模块
            attn_block = nn.Sequential(
                nn.Conv2d(in_ch * 2, in_ch, kernel_size=1),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(),
                nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch),
                nn.Sigmoid()
            )
            self.attn_convs.append(attn_block)
            self.lateral_convs.append(nn.Conv2d(in_ch, self.out_channels, kernel_size=1))
            self.smooth_convs.append(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1))

        self.initialized = True

    def forward(self, ms3, ms4, ms5, pan3, pan4, pan5):
        ms_feats = [ms3, ms4, ms5]
        pan_feats = [pan3, pan4, pan5]

        if not self.initialized:
            self._build_layers(ms_feats, pan_feats)
            self.attn_convs = self.attn_convs.to(ms3.device)
            self.lateral_convs = self.lateral_convs.to(ms3.device)
            self.smooth_convs = self.smooth_convs.to(ms3.device)

        fused_feats = []
        for i in range(len(ms_feats)):
            pan_feat = pan_feats[i]
            ms_feat = F.interpolate(ms_feats[i], size=pan_feat.shape[2:], mode='bilinear', align_corners=False)
            concat = torch.cat([ms_feat, pan_feat], dim=1)
            attn = self.attn_convs[i](concat)
            fused = ms_feat * attn + pan_feat * (1 - attn)
            fused_feats.append(fused)

        feats = [l_conv(f) for l_conv, f in zip(self.lateral_convs, fused_feats)]

        # 自顶向下 FPN 融合
        P5 = feats[2]
        P4 = feats[1] + F.interpolate(P5, size=feats[1].shape[2:], mode='nearest')
        P3 = feats[0] + F.interpolate(P4, size=feats[0].shape[2:], mode='nearest')

        P3 = self.smooth_convs[0](P3)
        P4 = self.smooth_convs[1](P4)
        P5 = self.smooth_convs[2](P5)

        return P3, P4, P5 