import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

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
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
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
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(3, 3), stride=1),
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
        l_n, c_n = 16,16
        self.Dictionary = nn.Parameter(
            torch.FloatTensor(l_n * c_n, 1, element_size * element_size * channel).to(torch.device("cuda:0")),
            requires_grad=True)
        nn.init.xavier_uniform_(self.Dictionary)
        self.unflod_win = nn.Unfold(kernel_size=(element_size, element_size), stride=element_size)
        self.flod_pan = nn.Fold(output_size=(64,64), kernel_size=(element_size, element_size), stride=element_size)
        self.CA = nn.MultiheadAttention(embed_dim=element_size * element_size * channel, num_heads=1, dropout=0)
        self.flod_win_1 = nn.Fold(output_size=(l_n * element_size, c_n * element_size),
                                  kernel_size=(element_size, element_size), stride=element_size)
        self.modality_classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # 将特征压成 [B, C, 1, 1]
                nn.Flatten(),                  # [B, C]
                nn.Linear(64, 2)              # 二分类：0 表示 MS，1 表示 PAN
                )
        
    def forward(self, x):
        size = x.size()
        D = self.Dictionary.repeat(1, size[0], 1) 
        x_w = self.unflod_win(x).permute(2, 0, 1) 

        q = x_w
        k = D
        v = D
        a = self.CA(q, k, v)[0] 
        
        representation = self.flod_pan(a.permute(1, 2, 0)) + x 
        visible_D = self.flod_win_1(self.Dictionary.permute(1, 2, 0)) 
        logits_pan = self.modality_classifier(representation)
        return logits_pan, representation, visible_D

class DictionaryRepresentationModule_m(nn.Module):
    def __init__(self):
        super(DictionaryRepresentationModule_m, self).__init__()
        element_size = 4
        channel = 64
        l_n, c_n = 4,4
        self.Dictionary = nn.Parameter(
            torch.FloatTensor(l_n * c_n, 1, element_size * element_size * channel).to(torch.device("cuda:0")),
            requires_grad=True)
        nn.init.xavier_uniform_(self.Dictionary)
        self.unflod_win = nn.Unfold(kernel_size=(element_size, element_size), stride=element_size)
        self.flod_ms = nn.Fold(output_size=(16,16), kernel_size=(element_size, element_size), stride=element_size)
        self.CA = nn.MultiheadAttention(embed_dim=element_size * element_size * channel, num_heads=1, dropout=0)
        self.flod_win_1 = nn.Fold(output_size=(l_n * element_size, c_n * element_size),
                                  kernel_size=(element_size, element_size), stride=element_size)
        self.modality_classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # 将特征压成 [B, C, 1, 1]
                nn.Flatten(),                  # [B, C]
                nn.Linear(64, 2)              # 二分类：0 表示 MS，1 表示 PAN
                )
        
    def forward(self, x):
        size = x.size()
        D = self.Dictionary.repeat(1, size[0], 1) # torch.Size([256, 128, 128])
        x_w = self.unflod_win(x).permute(2, 0, 1) # torch.Size([256, 128, 128])

        q = x_w
        k = D
        v = D
        a = self.CA(q, k, v)[0] #torch.Size([256, 128, 128])
        
        representation = self.flod_ms(a.permute(1, 2, 0))+x # torch.Size([128, 8, 64, 64])
        visible_D = self.flod_win_1(self.Dictionary.permute(1, 2, 0)) # torch.Size([1, 8, 64, 64])
        logits_ms = self.modality_classifier(representation)
        return logits_ms, representation, visible_D

class FE(nn.Module):
    def __init__(self, in_channels, feature_dim=128):
        super(FE, self).__init__()
        resnet = resnet50(pretrained=False)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu,
        )
        self.layer1 = resnet.layer1 
        self.layer2 = resnet.layer2 
        self.layer3 = resnet.layer3 
        self.layer4 = resnet.layer4  
        self.avgpool = resnet.avgpool

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
    '''
    def _make_projection_head4(self, in_dim, feature_dim):
        return nn.Sequential(
            nn.Linear(in_dim, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True)
        )
    def _make_projection_head3(self, in_dim, feature_dim):
        return nn.Sequential(
            nn.Linear(in_dim, feature_dim, bias=False),)
    '''
    def forward_stem(self,x):
        x_1 = self.stem(x) #[128,64,16,16] [128,64,64,64]
        return x_1

    def forward_layer12(self, x):
        x=self.layer1(x)# [128,256,16,16] [128,256,64,64]
        feat3_1 = self.layer2(x)  # conv3_x [128,512,8,8] [128,512,32,32]
        feat3_2 = self.avgpool(feat3_1) #[128,512,1,1] 
        feat3 = torch.flatten(feat3_2, start_dim=1)
        out3 = F.normalize(self.proj_head3(feat3), dim=-1) #[128,128]
        return out3,feat3_1
    
    def forward_layer3(self,x):
        feat4_1 = self.layer3(x)  # conv4_x [128,1024,4,4] [128,1024,16,16]
        feat4_2 = self.avgpool(feat4_1) #[128,1024,1,1]
        feat4 = torch.flatten(feat4_2, start_dim=1)
        out4 = F.normalize(self.proj_head4(feat4), dim=-1) #[128,128]
        return out4,feat4_1
    
    def forward(self,x):
        feat5_1 = self.layer4(x)  # conv5_x [128,2048,2,2] [128,2048,8,8]
        feat5_2 = self.avgpool(feat5_1) #[128,2048,1,1]
        feat5 = torch.flatten(feat5_2, start_dim=1)
        out5 = F.normalize(self.proj_head5(feat5), dim=-1) #[128,128]
        return out5, feat5_1

# PAN 模型
class PAN_Model(FE):
    def __init__(self, feature_dim=128):
        super(PAN_Model, self).__init__(in_channels=1, feature_dim=feature_dim)

# MS 模型
class MS_Model(FE):
    def __init__(self, feature_dim=128):
        super(MS_Model, self).__init__(in_channels=4, feature_dim=feature_dim)

class ResidualGate(nn.Module):
    def __init__(self, feat_dim=128, reduction=16):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feat_dim*2, feat_dim // reduction),
            nn.ReLU(),
            nn.Linear(feat_dim // reduction, feat_dim),
            nn.Sigmoid()
        )
    def forward(self,feat , r):  # r 是残差向量
        x = torch.cat([feat, r], dim=1)  
        return self.gate(x)  # 输出控制因子
    
class AlignGRU(nn.Module):
    def __init__(self, feature_dim=128, num_layers=1):
        super(AlignGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=feature_dim,
            num_layers=num_layers,
            batch_first=True
        )
    def forward(self, r_list):
        assert isinstance(r_list, list) and len(r_list) >= 2, "r_list must be a list of [B, C] tensors."
        r_seq = torch.stack(r_list, dim=1)  # [B, T, C]
        _, h_n = self.gru(r_seq)  # h_n: [1, B, C]
        r_dynamic = F.normalize(h_n.squeeze(0), dim=1)  # [B, C]
        return r_dynamic