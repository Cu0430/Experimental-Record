import sys 
sys.path.append('/home/aistudio/work/ex')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from sklearn.cluster import KMeans

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, feature_dim=128):
        super(FeatureExtractor, self).__init__()
        
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
        x = self.stem(x) #[128,64,4,4] [128,64,16,16]
        x = self.layer1(x)  # conv2_x [128,256,4,4] [128,256,16,16]
        feat3 = self.layer2(x)  # conv3_x [128,512,2,2] [128,512,8,8]
        feat4 = self.layer3(feat3)  # conv4_x [128,1024,1,1] [128,1024,4,4]
        feat5_1 = self.layer4(feat4)  # conv5_x [128,2048,1,1] [128,2048,2,2]
        
        feat3 = F.adaptive_avg_pool2d(feat3, (1, 1)) 
        feat4 = F.adaptive_avg_pool2d(feat4, (1, 1))
        feat5_2 = F.adaptive_avg_pool2d(feat5_1, (1, 1))

        # 展平特征  3:[128,512] 4:[128,1024] 5:[128,2048]
        feat3 = torch.flatten(feat3, start_dim=1)
        feat4 = torch.flatten(feat4, start_dim=1)
        feat5 = torch.flatten(feat5_2, start_dim=1)
        
        # 通过投影头
        out3 = F.normalize(self.proj_head3(feat3), dim=-1) #[128,128]
        out4 = F.normalize(self.proj_head4(feat4), dim=-1) #[128,128]
        out5 = F.normalize(self.proj_head5(feat5), dim=-1) #[128,128]
        
        return out3, out4, out5, feat5_2

# PAN 模型（1通道输入）
class PAN_Model(FeatureExtractor):
    def __init__(self, feature_dim=128):
        super(PAN_Model, self).__init__(in_channels=1, feature_dim=feature_dim)

# MS 模型（4通道输入）
class MS_Model(FeatureExtractor):
    def __init__(self, feature_dim=128):
        super(MS_Model, self).__init__(in_channels=4, feature_dim=feature_dim)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # ================= 特征增强模块 =================
        self.enhance_block = nn.Sequential(
            # 通道压缩 + 空间增强
            nn.Conv2d(2048, 1024, kernel_size=1),       # [B,1024,1,1]
            nn.BatchNorm2d(1024),
            nn.GELU(),
            
            # 深度特征交互
            nn.Conv2d(1024, 512, kernel_size=1),        # [B,512,1,1]
            nn.BatchNorm2d(512),
            nn.GELU(),
            
            nn.Conv2d(512, 512, kernel_size=1),       
            nn.BatchNorm2d(512),
            nn.GELU(),
            
            nn.Conv2d(512,1024, kernel_size=1),        
            nn.BatchNorm2d(1024),
            nn.GELU(),
            
            nn.Conv2d(1024, 2048, kernel_size=1),        # [B,2048,1,1]
            nn.BatchNorm2d(2048),
            nn.GELU(),
            
            # 通道注意力
            ChannelAttention(2048)                       # 保持维度
        )
        
        # ================= 解码模块 =================
        self.decoder = nn.Sequential(
            # 阶段1: 1x1 → 2*2 (使用转置卷积)
            nn.ConvTranspose2d(2048,1024, kernel_size=2, stride=2),  # [B,1024,2,2]
            nn.BatchNorm2d(1024),
            nn.GELU(),
            
            self._upsample_block(1024,512, scale=2),      #[B,512,4,4]
            self._upsample_block(512,256, scale=2),      #[B,256,8,8]
            # 阶段2: 4x4 → 16x16
            self._upsample_block(256, 128, scale=2),    # [B,128,16,16]
            
            # 阶段3: 16x16 → 64x64 
            self._upsample_block(128, 64, scale=4),     # [B,64,64,64]
            
            # 最终输出层
            nn.Conv2d(64, 4, kernel_size=3, padding=1), # [B,4,64,64]
            nn.Tanh()                                    # 限制输出范围
        )

    def _upsample_block(self, in_c, out_c, scale):
        """上采样模块 (混合插值和卷积)"""
        return nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.GELU()
        )

    def forward(self, x):
        if torch.isnan(x).any():
            print("Decoder输入包含NaN值!")
            print(x)
        # 输入验证
        assert x.shape[-2:] == (1,1), f"要求输入空间尺寸为1x1, 实际输入 {x.shape}"
        
        # 特征增强
        enhanced = self.enhance_block(x)                # [B,2048,1,1]
        
        # 解码过程
        decoded = self.decoder(enhanced)                # [B,4,64,64]
        return decoded,enhanced

class ChannelAttention(nn.Module):
    """轻量化通道注意力"""
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        if torch.isnan(x).any():
            print("注意力输入异常!")
            print(x)
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class DictionaryRepresentationModule1(nn.Module):
    def __init__(self, feature_dim=2048, num_patches=8, dict_size=256, embed_dim=256):
        super(DictionaryRepresentationModule1, self).__init__()
        
        self.feature_dim = feature_dim  # 输入特征维度
        self.num_patches = num_patches  # 分块数量
        self.embed_dim = embed_dim  # 降维后的维度（改为 `feature_dim // num_patches` 以保证输出不变）
        self.dict_size = dict_size  # 字典大小

        # 线性投影层，将高维特征降维成 `embed_dim`
        self.proj = nn.Linear(feature_dim // num_patches, embed_dim)

        # 定义可学习字典
        self.Dictionary = nn.Parameter(torch.FloatTensor(dict_size, embed_dim))
        nn.init.xavier_uniform_(self.Dictionary)  
        self.norm=nn.LayerNorm(embed_dim)
        
        # 注意力机制
        self.CA = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=0)

        # 还原特征维度
        self.restore_proj = nn.Linear(num_patches * embed_dim, feature_dim)

    def forward(self, x):
        batch_size, feature_dim, _, _ = x.shape  # [128, 2048, 1, 1]

        # 变换输入形状
        x = x.squeeze(-1).squeeze(-1)  # [128, 2048]

        # 将 `2048` 维特征分成 `num_patches=8` 份，每份 `2048/8=256`
        x_patches = x.view(batch_size, self.num_patches, feature_dim // self.num_patches)  # [128, 8, 256]
        x_patches = self.proj(x_patches)  # 线性变换至 [128, 8, 256]
        x_patches = self.norm(x_patches)
        x_patches = x_patches.permute(1, 0, 2)  # 变成 [8, 128, 256]，适应 `MultiheadAttention`

        # 扩展字典维度
        D = self.Dictionary.unsqueeze(1).repeat(1, batch_size, 1)  # [256, 128, 256]

        # 计算注意力
        q = x_patches  # [8, 128, 256]
        k = D  # [256, 128, 256]
        v = D  # [256, 128, 256]
        
        attn_output = self.CA(q, k, v)[0]  # [8, 128, 256]

        # 重新调整形状
        representation = attn_output.permute(1, 0, 2).reshape(batch_size, -1)  # [128, 8*256] = [128, 2048]

        # **恢复形状**（增加 `1,1` 维度）
        representation = representation.unsqueeze(-1).unsqueeze(-1)  # [128, 2048, 1, 1]
        
        representation = torch.tanh(representation)  # 限制输出到[-1,1]
        return representation  # **保证输出与输入一致**

class DictionaryRepresentationModule(nn.Module):
    def __init__(self):
        super(DictionaryRepresentationModule, self).__init__()
        element_size = 3
        channel = 2048
        l_n = 16
        c_n = 16
        self.Dictionary = nn.Parameter(
            torch.FloatTensor(l_n * c_n, 1, element_size * element_size * channel).to(torch.device("cuda:0")),
            requires_grad=True)
        nn.init.uniform_(self.Dictionary, 0, 1)
        self.unflod_win = nn.Unfold(kernel_size=(element_size, element_size), stride=element_size)
        self.flod_win = nn.Fold(output_size=(64,64), kernel_size=(element_size, element_size), stride=element_size)
        self.CA = nn.MultiheadAttention(embed_dim=element_size * element_size * channel, num_heads=1, dropout=0)
        self.flod_win_1 = nn.Fold(output_size=(l_n * element_size, c_n * element_size),
                                  kernel_size=(element_size, element_size), stride=element_size)

    def forward(self, x):
        size = x.size()
        D = self.Dictionary.repeat(1, size[0], 1)
        x_w = self.unflod_win(x).permute(2, 0, 1)

        q = x_w
        k = D
        v = D
        a = self.CA(q, k, v)[0]

        representation = self.flod_win(a.permute(1, 2, 0))
        visible_D = self.flod_win_1(self.Dictionary.permute(1, 2, 0))

        return representation, visible_D






