import torch
from torchvision.utils import make_grid
from loss2 import ContrastiveLoss,IncrementalFalseNegativeDetection,L_Grad,CorrelationCoefficient,Loss_intensity
import torch.nn.functional as F
import yaml # 用于解析配置文件 config.
import model2
import numpy as np # 用于数值计算
import cv2 # 计算机视觉库，用于图像处理
from tifffile import imread # 用于读取 .tif 格式的遥感图像
from dataset2 import MyData1, _create_model_training_folder # 自定义数据集类
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts # 余弦退火学习率调度器
import os
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter # SummaryWriter：用于记录训练过程中的信息，方便在TensorBoard中可视化
from tqdm import tqdm # 用于显示进度条的库
import torch.nn as nn
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

os.environ["OMP_NUM_THREADS"] = "1" # 设置 OpenMP 线程数，避免多线程冲突
# print(torch.__version__)torch.manual_seed(0) # 设置随机种子，保证实验可复现
#seed = 42
#torch.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)
#np.random.seed(seed)
#random.seed(seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False 

class Trainer:
    def __init__(self, FE_ms, FE_pan, decoder,MSDP, PANDP, gate3,gate4,align_gru,
                 optimizer, scheduler, device, **params):
        
        self.FE_ms = FE_ms
        self.FE_pan = FE_pan
        self.decoder = decoder
        self.MSDP = MSDP
        self.PANDP = PANDP
        self.gate3 = gate3
        self.gate4=gate4
        self.align_gru=align_gru
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        self.max_epochs = params['max_epochs']
        self.writer = SummaryWriter()
        self.batch_size = params['batch_size']
        self.num_workers = 0
        self.checkpoint_interval = params['checkpoint_interval']
        self.temp = params['temp']
        _create_model_training_folder(self.writer, files_to_same=["/root/autodl-tmp/config/config.yaml"])
        

    def _visualize_stage(self, ms_feat, pan_feat, tag, niter, ms_scale=4):
        """
        统一特征可视化方法
        :param ms_feat: MS特征 [B,C,H,W]
        :param pan_feat: PAN特征 [B,C,H,W]
        :param tag: 阶段标识
        :param niter: 当前迭代次数
        :param ms_scale: MS特征需要上采样的倍数
        """
        with torch.no_grad():
            # 取第一个样本
            ms_sample = ms_feat[0].unsqueeze(0)
            pan_sample = pan_feat[0].unsqueeze(0)

            # 统一处理方法
            def prepare(feat, scale=1):
                feat = F.interpolate(feat, scale_factor=scale, mode='nearest')
                feat = feat[:,:3]  # 取前3通道
                feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-6)
                return feat

            # 处理MS特征（上采样指定倍数）
            ms_vis = prepare(ms_sample, scale=ms_scale)
            # 处理PAN特征（保持原尺寸）
            pan_vis = prepare(pan_sample)

            # 创建对比网格
            grid = torch.cat([ms_vis, pan_vis], dim=3)
            grid = make_grid(grid, nrow=1, normalize=False)
            self.writer.add_image(f"Alignment/{tag}", grid, niter)

    def visualize_alignment_tsne(self, ms_feat, pan_feat, tag, niter):
        """
        可视化模态对齐特征分布（t-SNE 降维）
        :param ms_feat: [B, D]
        :param pan_feat: [B, D]
        """
        with torch.no_grad():
            ms_feat_np = ms_feat.detach().cpu().numpy()
            pan_feat_np = pan_feat.detach().cpu().numpy()

            # 构造 t-SNE 输入
            all_feat = np.concatenate([ms_feat_np, pan_feat_np], axis=0)
            labels = np.array([0]*len(ms_feat_np) + [1]*len(pan_feat_np))  # 0:MS, 1:PAN

            # 降维
            tsne = TSNE(n_components=2, random_state=0, perplexity=10)
            reduced_feat = tsne.fit_transform(all_feat)

            # 可视化
            plt.figure(figsize=(6, 6))
            plt.scatter(reduced_feat[labels == 0, 0], reduced_feat[labels == 0, 1], c='blue', label='MS', alpha=0.6)
            plt.scatter(reduced_feat[labels == 1, 0], reduced_feat[labels == 1, 1], c='green', label='PAN', alpha=0.6)
            plt.legend()
            plt.title(f'Modal Alignment - {tag}')
            plt.grid(True)

            # 保存或写入 TensorBoard
            self.writer.add_figure(f"TSNE_Alignment/{tag}", plt.gcf(), niter)
            plt.close()


    def train(self, train_dataset):
        # drop_last=False 表示如果数据集大小不能被批次大小整除，不丢弃最后一个批次
        # shuffle=True 表示每个 epoch 训练前对数据进行打乱
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)
        Lgrad = L_Grad()
        CC = CorrelationCoefficient()
        contrastive_loss = ContrastiveLoss(temperature=0.07, negative_weight=0.8,num_clusters=100, acceptance_rate=0.1)
        fn_detector = IncrementalFalseNegativeDetection(max_clusters=50, acceptance_rate=0.1)
        criterion_modality = nn.CrossEntropyLoss()

        niter = 0 # 初始化迭代次数

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        iters = self.max_epochs * 209557 // self.batch_size # 迭代次数=epoch*每个epoch的批次数
        

        for epoch_counter in range(self.max_epochs):
            epoch_loss_Contrastive=[]
            epoch_loss_MSDP = []
            epoch_loss_PANDP = []
            epoch_loss_same=[]
            epoch_loss_consistency=[]
            epoch_loss_diversity=[]
            total_num, total_loss, train_bar = 0, 0, tqdm(train_loader)

            for idx, (ms, pan, _, _) in enumerate(train_bar):                
                #enumerate在迭代时同时获取索引和值，默认从0开始
                ms = ms.to(self.device) #[128,4,16,16]
                pan = pan.to(self.device) #[128,1,64,64]

                if niter == 0:
# 将批次中的前 32 张图像通过 make_grid 转换为网格图像并写入 TensorBoard，以便在训练过程中可视化图像
                    grid = torchvision.utils.make_grid(ms[:32])
                    self.writer.add_image('ms', grid, global_step=niter)

                    grid = torchvision.utils.make_grid(pan[:32])
                    self.writer.add_image('pan', grid, global_step=niter)
                
                ms1=self.FE_ms.forward_stem(ms) #[128,64,16,16]
                pan1=self.FE_pan.forward_stem(pan) #[128,64,64,64]
                
                ms2_up = F.interpolate(ms1, size=(64, 64), mode='bilinear', align_corners=True)
                simple_fusion=ms2_up+pan1
                fusion_image,fusion_f=self.decoder(simple_fusion)#[128,4,64,64][128,64,64,64]
                
                logits_ms,MSDP_ms,_=self.MSDP(ms1) 
                logits_pan,PANDP_pan,_=self.PANDP(pan1)
                # 创建模态标签
                modality_label_ms = torch.zeros(ms.size(0), dtype=torch.long).to(self.device)  # MS 为 0
                modality_label_pan = torch.ones(pan.size(0), dtype=torch.long).to(self.device)  # PAN 为 1

                ms3,ms3_1=self.FE_ms.forward_layer12(MSDP_ms)
                pan3,pan3_1=self.FE_pan.forward_layer12(PANDP_pan)
                r3 = F.normalize(ms3 - pan3, dim=1)
                               
                ms4,ms4_1=self.FE_ms.forward_layer3(ms3_1)
                pan4,pan4_1=self.FE_pan.forward_layer3(pan3_1)
                g3 = self.gate3(0.5 * (ms4 + pan4), r3) 
                ms4_mod = F.normalize(ms4 - g3 * r3, dim=1)
                pan4_mod = F.normalize(pan4 + g3 * r3, dim=1)
                r4 = F.normalize(ms4_mod - pan4_mod, dim=1)
                r4_1 = self.align_gru([r3, r4])
                
                ms5,ms5_1=self.FE_ms(ms4_1)
                pan5,pan5_1=self.FE_pan(pan4_1)
                g4=self.gate4(0.5 * (ms5 + pan5),r4_1)
                ms5_mod=F.normalize(ms5-g4*r4_1, dim=1)
                pan5_mod=F.normalize(pan5+g4*r4_1, dim=1)
                r5=F.normalize(ms5_mod-pan5_mod, dim=1)
                                              
                if niter % 500 == 0:
                    self._visualize_stage(ms1, pan1, "01_stem", niter, ms_scale=4)
                    #self._visualize_stage(ms2, pan2, "02_Layer1", niter, ms_scale=4)
                    self._visualize_stage(MSDP_ms, PANDP_pan, "03_Dictionary", niter, ms_scale=4)
                    self._visualize_stage(ms3_1, pan3_1, "04_Layer2", niter, ms_scale=4)
                    self._visualize_stage(ms4_1, pan4_1, "05_Layer3", niter, ms_scale=4)
                    self._visualize_stage(ms5_1, pan5_1, "06_Layer4", niter, ms_scale=4)
                    # 各层次特征
                    self.visualize_alignment_tsne(ms3, pan3, "04_Layer2", niter)
                    self.visualize_alignment_tsne(ms4_mod, pan4_mod, "05_Layer3", niter)
                    self.visualize_alignment_tsne(ms5_mod, pan5_mod, "06_Layer4", niter)
                T=(epoch_counter+1)/self.max_epochs
                w1 = (1 - T) * 0.3      # 前期注重低层
                w2 = 0.3
                w3 = T * 0.4 + 0.3                             
                loss1 = contrastive_loss(ms3, pan3)                               
                loss2 = contrastive_loss(ms4_mod, pan4_mod)               
                loss3 = contrastive_loss(ms5_mod, pan5_mod)

                sim_r34 = F.cosine_similarity(r3.detach(), r4, dim=1).mean()
                sim_r45 = F.cosine_similarity(r4.detach(), r5, dim=1).mean()

                loss_consistency = -sim_r34-sim_r45
                lambda_consistency = T**1.25 * 0.1  # 可调，逐渐增大以辅助收敛
                
                # 鼓励它们不太相似，如果超过 0.7 则惩罚
                loss_diversity = F.relu(sim_r34 - 0.7) + F.relu(sim_r45 - 0.7)
                lambda_diversity = T * 0.1  # 同样动态调节，控制差异性惩罚权重                            
                loss_Contrastive= w1*loss1+w2*loss2+w3*loss3
                
                loss_fusion = Lgrad(ms, pan, fusion_image) + Loss_intensity(ms, pan, fusion_image)
                loss_MSDP = -CC(MSDP_ms, fusion_f.detach())#[128,256,4,4] [128,256,16,16]
                loss_PANDP = -CC(PANDP_pan, fusion_f.detach())#[128,256,16,16]
                                             
                # 计算模态分类损失
                loss_modal_ms = criterion_modality(logits_ms, modality_label_ms)
                loss_modal_pan = criterion_modality(logits_pan, modality_label_pan)
                loss_modal = loss_modal_ms + loss_modal_pan
                
                loss = loss_Contrastive + loss_fusion + loss_MSDP + loss_PANDP + loss_modal+lambda_consistency * loss_consistency - lambda_diversity * loss_diversity

                self.optimizer.zero_grad() # 清零梯度
                loss.backward() # 反向传播
                self.optimizer.step() # 更新模型参数
                self.scheduler.step(self.max_epochs + idx / iters) # 更新学习率调度器
                
                epoch_loss_Contrastive.append(loss_Contrastive.item())
                epoch_loss_MSDP.append(loss_MSDP.item())
                epoch_loss_PANDP.append(loss_PANDP.item())
                epoch_loss_same.append(loss_modal.item())
                epoch_loss_consistency.append(loss_consistency.item())
                epoch_loss_diversity.append(loss_diversity.item())

                niter += 1
                total_num += self.batch_size
                total_loss += loss.item() * self.batch_size
                train_bar.set_description(
                    'Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch_counter+1, self.max_epochs, total_loss / total_num))

                self.writer.add_scalar('total loss', total_loss / total_num, global_step=niter)
                self.writer.add_scalar('fusion loss', loss_fusion, global_step=niter)
                self.writer.add_scalar('MSDP loss', loss_MSDP, global_step=niter)
                self.writer.add_scalar('contrastive loss', loss_Contrastive, global_step=niter)
                self.writer.add_scalar('PANDP loss', loss_PANDP, global_step=niter)
                self.writer.add_scalar('modal loss', loss_modal, global_step=niter)
                self.writer.add_scalar('consistency loss', loss_consistency, global_step=niter)
                self.writer.add_scalar('diversity loss', loss_diversity, global_step=niter)
            #fn_detector.update_acceptance_rate(epoch_counter+1, self.max_epochs)
            #fn_detector.update_clusters(epoch_counter+1, self.max_epochs)
            print("End of epoch {}".format(epoch_counter+1))
            epoch_loss_Contrastive_mean = np.mean(epoch_loss_Contrastive)
            epoch_loss_MSDP_mean = np.mean(epoch_loss_MSDP)
            epoch_loss_PANDP_mean = np.mean(epoch_loss_PANDP)
            epoch_loss_same_mean = np.mean(epoch_loss_same)
            epoch_loss_diversity_mean = np.mean(epoch_loss_diversity)
            epoch_loss_consistency_mean = np.mean(epoch_loss_consistency)
            print(" -loss_Contrastive " + str(epoch_loss_Contrastive_mean))
            print(" -loss_MSDP " + str(epoch_loss_MSDP_mean) + " -loss_PANDP " + str(
                  epoch_loss_PANDP_mean))
            print(" -loss_modal " + str(epoch_loss_same_mean))
            print(" -loss_consistency " + str(epoch_loss_consistency_mean) + " -loss_diversity " + str(
                  epoch_loss_diversity_mean))
            print()
        self.save_model(os.path.join(model_checkpoints_folder, 'model.pth'))
        
    

    def save_model(self, PATH):

        torch.save({
            'FE_ms_state_dict': self.FE_ms.state_dict(),
            'FE_pan_state_dict': self.FE_pan.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'MSDP_state_dict': self.MSDP.state_dict(),
            'PANDP_state_dict': self.PANDP.state_dict(),
            'gate3_state_dict':self.gate3.state_dict(),
            'gate4_state_dict':self.gate4.state_dict(),
            'align_gru_state_dict':self.align_gru.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            
            }, PATH)


def my_main():

    Unlabel_Rate = 0.05 # 无标签数据比例（5%）
    # 读取 PAN（全色）图像
    pan_np = imread('/root/autodl-tmp/datasets/hhht/pan.tif')
    print('The shape of the original pan:', np.shape(pan_np))

    # 读取 MS（多光谱）图像
    ms4_np = imread('/root/autodl-tmp/datasets/hhht/ms4.tif')
    print('The shape of the original MS:', np.shape(ms4_np))

    # 读取标签数据（npy 格式）
    label_np = np.load("/root/autodl-tmp/datasets/hhht/train.npy")
    print('The shape of the train label：', np.shape(label_np))

    Ms4_patch_size = 16 # MS 图像的 patch 大小 16

    Interpolation = cv2.BORDER_REFLECT_101  # 边界填充方式 gfedcb|abcdefgh|gfedcba
    #对称法，以最边缘像素为轴，对称
    #扩充src边缘，将图像变大，便于处理边界
    
    # 对 MS 图像进行边界填充
    #在进行 patch 采样时，通常是以中心像素为基准，向四周扩展 patch_size / 2 的区域
    #但是，patch 大小是偶数（16），因此当 patch_size / 2 = 8 时，中心位置偏向左上角
    #对于左上角的像素 (0,0)，如果直接提取 16×16 补丁，会超出边界，所以必须填充边界来补足数据。
    top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2),
                                                    int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2))
    ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)
    print('The shape of the MS picture after padding', np.shape(ms4_np))

    #减4的目的：
    #确保 MS 和 PAN 图像的 patch 采样中心尽量对齐
    #避免 PAN 图像 patch 由于整数计算导致边界不齐
    #减少融合时的边界效应，提高配准精度
    Pan_patch_size = Ms4_patch_size*4  #64
    top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2),
                                                    int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2)) # 28，32
    pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)
    print('The shape of the PAN picture after padding', np.shape(pan_np))

    # label_np=label_np.astype(np.uint8)
    label_np = label_np - 1 # 使标签从0变为255

    # np.unique()：去重+从小到大排序
    # return_counts=true，返回去重数组中的元素在原数组中的出现次数
    label_element, element_count = np.unique(label_np, return_counts=True)
    print('Class label:', label_element)
    print('Number of samples in each category:', element_count)
    Categories_Number = len(label_element) - 1
    print('Number of categories labeled:', Categories_Number)
    label_row, label_column = np.shape(label_np)

    def to_tensor(image):
        max_i = np.max(image)
        min_i = np.min(image)
        image = (image - min_i) / (max_i - min_i)
        return image
    
    unlabeled_xy = []

    for row in range(label_row):  # 行
        for column in range(label_column):
            if label_np[row][column] == 255:
                unlabeled_xy.append([row, column])
    
    unlabeled_xy = np.array(unlabeled_xy)
    np.random.shuffle(unlabeled_xy)
    unlabeled_xy = unlabeled_xy[:int(len(unlabeled_xy) * Unlabel_Rate)]
    print("{} sets of unlabeled data are used".format(len(unlabeled_xy)))

    unlabeled_xy = torch.from_numpy(unlabeled_xy).type(torch.LongTensor)

    ms4 = to_tensor(ms4_np)
    pan = to_tensor(pan_np)
    pan = np.expand_dims(pan, axis=0) # 给 pan 数组在指定的轴（axis=0）上增加一个新的维度
    #使其形状从 (H, W) 变为 (1, H, W)，表示这是一个包含单张图像的数据集
    ms4 = np.array(ms4).transpose((2, 0, 1)) # ms4 数组的维度从 (H, W, C) 转置为 (C, H, W)

    ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
    pan = torch.from_numpy(pan).type(torch.FloatTensor)

    config = yaml.load(open("/root/autodl-tmp/config/config.yaml", "r"), Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    unlabeled_data = MyData1(ms4, pan, unlabeled_xy, Ms4_patch_size)
    
    FE_ms = model2.MS_Model().to(device)
    FE_pan = model2.PAN_Model().to(device)
    decoder = model2.Decoder().to(device)
    MSDP = model2.DictionaryRepresentationModule_m().to(device)
    PANDP = model2.DictionaryRepresentationModule_p().to(device)
    gate3 = model2.ResidualGate().to(device)
    gate4 = model2.ResidualGate().to(device)
    align_gru = model2.AlignGRU(feature_dim=128).to(device)

    optimizer = torch.optim.Adam(list(FE_ms.parameters()) + list(decoder.parameters()) +
                                list(FE_pan.parameters()) + list(MSDP.parameters()) + 
                                list(PANDP.parameters())+ list(gate3.parameters())+ 
                                list(gate4.parameters())+list(align_gru.parameters()), **config['optimizer']['params'])
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    #每 5 个 epoch 就进行一次学习率重启，使用余弦退火策略调整学习率

    trainer = Trainer(FE_ms=FE_ms, FE_pan=FE_pan, decoder=decoder, 
                         MSDP=MSDP, PANDP=PANDP, gate3=gate3,gate4=gate4,align_gru=align_gru,
                         optimizer=optimizer, scheduler=scheduler,device=device,**config['trainer']) 
                      
    trainer.train(unlabeled_data) # 开始训练，unlabeled_data 是训练的数据集，包含无标签数据

if __name__ == '__main__':
    my_main()