from loss import ContrastiveLoss,IncrementalFalseNegativeDetection,L_Grad,CorrelationCoefficient,Loss_intensity
import torch.nn.functional as F

import torch # 导入 PyTorch 进行深度学习计算
import yaml # 用于解析配置文件 config.
import model 
import numpy as np # 用于数值计算
import cv2 # 计算机视觉库，用于图像处理
from tifffile import imread # 用于读取 .tif 格式的遥感图像
from dataset import MyData1, _create_model_training_folder # 自定义数据集类
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts # 余弦退火学习率调度器

import os
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter # SummaryWriter：用于记录训练过程中的信息，方便在TensorBoard中可视化
from tqdm import tqdm # 用于显示进度条的库


os.environ["OMP_NUM_THREADS"] = "1" # 设置 OpenMP 线程数，避免多线程冲突
# print(torch.__version__)torch.manual_seed(0) # 设置随机种子，保证实验可复现


class Trainer:
    def __init__(self, FE_ms, FE_pan, decoder, 
                MSDP, PANDP, optimizer, scheduler, device, **params):
        self.FE_ms = FE_ms
        self.FE_pan = FE_pan
        self.decoder = decoder
        
        self.MSDP = MSDP
        self.PANDP = PANDP
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        self.max_epochs = params['max_epochs']
        self.writer = SummaryWriter()

        self.batch_size = params['batch_size']
        self.num_workers = 0
        self.checkpoint_interval = params['checkpoint_interval']
    
        self.temp = params['temp']
        _create_model_training_folder(self.writer, files_to_same=["./config/config.yaml", "./train.py"])

  

    def train(self, train_dataset):
        # drop_last=False 表示如果数据集大小不能被批次大小整除，不丢弃最后一个批次
        # shuffle=True 表示每个 epoch 训练前对数据进行打乱
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)
        Lgrad = L_Grad()
        CC = CorrelationCoefficient()
        contrastive_loss = ContrastiveLoss(temperature=0.07, negative_weight=0.8,num_clusters=100, acceptance_rate=0.1)
        fn_detector = IncrementalFalseNegativeDetection(num_clusters=100, acceptance_rate=0.1)
        
        niter = 0 # 初始化迭代次数

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        iters = self.max_epochs * 177596 // self.batch_size # 迭代次数=epoch*每个epoch的批次数
        
        for epoch_counter in range(self.max_epochs):
            epoch_loss_Contrastive=[]
            epoch_loss_MSDP = []
            epoch_loss_PANDP = []
            epoch_loss_same=[]
            total_num, total_loss, train_bar = 0, 0, tqdm(train_loader)

            for idx, (ms, pan, _, _) in enumerate(train_bar):
                if torch.isnan(ms).any() or torch.isnan(pan).any():
                    print("NaN in input data!")
                    continue
                
                #enumerate在迭代时同时获取索引和值，默认从0开始
                ms = ms.to(self.device) #[128,4,16,16]
                pan = pan.to(self.device) #[128,1,64,64]

                if niter == 0:
# 将批次中的前 32 张图像通过 make_grid 转换为网格图像并写入 TensorBoard，以便在训练过程中可视化图像
                    grid = torchvision.utils.make_grid(ms[:32])
                    self.writer.add_image('ms', grid, global_step=niter)

                    grid = torchvision.utils.make_grid(pan[:32])
                    self.writer.add_image('pan', grid, global_step=niter)

                ms3,ms4,ms5,ms_f=self.FE_ms(ms) #[128,2048,1,1]
                
                pan3,pan4,pan5,pan_f=self.FE_pan(pan)
                
                simple_fusion=ms_f+pan_f
                fusion_image,fusion_f=self.decoder(simple_fusion)  #fusion_f=[128,2048,1,1]                  
                
                
                MSDP_ms=self.MSDP(ms_f) 
                PANDP_pan=self.PANDP(pan_f)  
                
                #ms3 = F.normalize(ms3, dim=1)  # 归一化
                #pan3 = F.normalize(ms3, dim=1)
                  
                
                loss1 = contrastive_loss(ms3, pan3)
                                
                loss2 = contrastive_loss(ms4, pan4)
                
                loss3 = contrastive_loss(ms5, pan5)
                              
                loss_Contrastive= 0.25*loss1+0.25*loss2+0.5*loss3
                
                loss_fusion = Lgrad(ms, pan, fusion_image) + Loss_intensity(ms, pan, fusion_image)
                loss_MSDP = -CC(MSDP_ms, fusion_f.detach())
                loss_PANDP = -CC(PANDP_pan, fusion_f.detach())
                loss_same = F.mse_loss(MSDP_ms, PANDP_pan)
                loss = loss_Contrastive + loss_fusion + loss_MSDP + loss_PANDP + 0.5*loss_same
                
                self.optimizer.zero_grad() # 清零梯度
                loss.backward() # 反向传播
                self.optimizer.step() # 更新模型参数
                self.scheduler.step(self.max_epochs + idx / iters) # 更新学习率调度器
                
                epoch_loss_Contrastive.append(loss_Contrastive.item())
                epoch_loss_MSDP.append(loss_MSDP.item())
                epoch_loss_PANDP.append(loss_PANDP.item())
                epoch_loss_same.append(loss_same.item())
                
                niter += 1
                if niter % 50000 == 0:  # 每 500 次迭代记录一次
                    msdp_features = MSDP_ms.detach().cpu().numpy().reshape(len(MSDP_ms), -1)
                    pandp_features = PANDP_pan.detach().cpu().numpy().reshape(len(PANDP_pan), -1)
                    ms_features = ms_f.detach().cpu().numpy().reshape(len(ms_f), -1)
                    pan_features = pan_f.detach().cpu().numpy().reshape(len(pan_f), -1)
                    fusion_features = fusion_f.detach().cpu().numpy().reshape(len(fusion_f), -1)
                       
                    self.writer.add_embedding(msdp_features, global_step=niter, tag="MSDP_ms")
                    self.writer.add_embedding(pandp_features,  global_step=niter, tag="PANDP_pan")
                    self.writer.add_embedding(ms_features,  global_step=niter, tag="ms_f")
                    self.writer.add_embedding(pan_features,  global_step=niter, tag="pan_f")
                    self.writer.add_embedding(fusion_features,  global_step=niter, tag="fusion_f")


                total_num += self.batch_size
                total_loss += loss.item() * self.batch_size
                train_bar.set_description(
                    'Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch_counter+1, self.max_epochs, total_loss / total_num))
                self.writer.add_scalar('total loss', total_loss / total_num, global_step=niter)
            fn_detector.update_acceptance_rate(epoch_counter+1, self.max_epochs)
            print("End of epoch {}".format(epoch_counter+1))
            epoch_loss_Contrastive_mean = np.mean(epoch_loss_Contrastive)
            epoch_loss_MSDP_mean = np.mean(epoch_loss_MSDP)
            epoch_loss_PANDP_mean = np.mean(epoch_loss_PANDP)
            epoch_loss_same_mean = np.mean(epoch_loss_same)

            print()
            print(" -epoch " + str(epoch_counter+1))
            print(" -loss_Contrastive " + str(epoch_loss_Contrastive_mean))
            print(" -loss_MSDP " + str(epoch_loss_MSDP_mean) + " -loss_PANDP " + str(
                  epoch_loss_PANDP_mean))
            print(" -loss_same " + str(epoch_loss_same_mean))
            print()
        self.save_model(os.path.join(model_checkpoints_folder, 'model.pth'))
        
    

    def save_model(self, PATH):

        torch.save({
            'FE_ms_state_dict': self.FE_ms.state_dict(),
            'FE_pan_state_dict': self.FE_pan.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'MSDP_state_dict': self.MSDP.state_dict(),
            'PANDP_state_dict': self.PANDP.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, PATH)


def my_main():

    Train_Rate = 0.05 # 训练集比例（5%）
    Unlabel_Rate = 0.05 # 无标签数据比例（5%）
    # 读取 PAN（全色）图像
    pan_np = imread('./datasets/hhht/pan.tif')
    print('The shape of the original pan:', np.shape(pan_np))

    # 读取 MS（多光谱）图像
    ms4_np = imread('./datasets/hhht/ms4.tif')
    
    print('The shape of the original MS:', np.shape(ms4_np))

    # 读取标签数据（npy 格式）
    label_np = np.load("./datasets/hhht/label.npy")
    print('The shape of the label：', np.shape(label_np))

    Ms4_patch_size = 16 # MS 图像的 patch 大小

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
    Pan_patch_size = Ms4_patch_size * 4 #64
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
    
    
    
    # 处理有标签和无标签数据
    # arange（最小值，最大值，步长）（左闭右开） : 创建等差数列
    ground_xy = np.array([[]] * Categories_Number).tolist() #.tolist()：将其转换回 Python 列表
    ground_xy_allData = np.arange(label_row * label_column * 2).reshape(label_row * label_column,
                                                                        2)
    unlabeled_xy = []

    count = 0
    for row in range(label_row):  # 行
        for column in range(label_column):
            ground_xy_allData[count] = [row, column]
            count = count + 1
            if label_np[row][column] != 255:
                # 如果该像素点不是无标签点（即标签不是 255）
                ground_xy[int(label_np[row][column])].append([row, column])
            else:
                unlabeled_xy.append([row, column])

    length_unlabel = len(unlabeled_xy) # 3551937
    using_length = length_unlabel * Unlabel_Rate
    unlabeled_xy = unlabeled_xy[0:int(using_length)]
    print("{} sets of unlabeled data are used".format(len(unlabeled_xy)))

    for categories in range(Categories_Number):
        ground_xy[categories] = np.array(ground_xy[categories])
        # 将每个类别的 ground_xy 数据转换为 NumPy 数组，确保是数组形式
        shuffle_array = np.arange(0, len(ground_xy[categories]), 1)
        np.random.shuffle(shuffle_array) # 打乱 shuffle_array 数组的顺序
        ground_xy[categories] = ground_xy[categories][shuffle_array] # 按打乱后的索引重新排列数据
    
    # 对所有数据进行统一打乱
    shuffle_array = np.arange(0, label_row * label_column, 1)
    np.random.shuffle(shuffle_array)
    ground_xy_allData = ground_xy_allData[shuffle_array]
    unlabeled_xy = np.array(unlabeled_xy)
    
    # 构造训练集和测试集
    ground_xy_train = []
    ground_xy_test = []
    label_train = []
    label_test = []

    for categories in range(Categories_Number):
        categories_number = len(ground_xy[categories])
        # print('aaa', categories_number)
        for i in range(categories_number):
            if i < int(categories_number * Train_Rate):
                ground_xy_train.append(ground_xy[categories][i])
            else:
                ground_xy_test.append(ground_xy[categories][i])
        label_train = label_train + [categories for x in range(int(categories_number * Train_Rate))]
        label_test = label_test + [categories for x in range(categories_number - int(categories_number * Train_Rate))]

    label_train = np.array(label_train)
    label_test = np.array(label_test)
    ground_xy_train = np.array(ground_xy_train)
    ground_xy_test = np.array(ground_xy_test)

    # 打乱测试集和训练集
    shuffle_array = np.arange(0, len(label_test), 1)
    np.random.shuffle(shuffle_array)
    label_test = label_test[shuffle_array]
    ground_xy_test = ground_xy_test[shuffle_array]

    shuffle_array = np.arange(0, len(label_train), 1)
    np.random.shuffle(shuffle_array)
    label_train = label_train[shuffle_array]
    ground_xy_train = ground_xy_train[shuffle_array]


    label_train = torch.from_numpy(label_train).type(torch.LongTensor) #longtensor是64位整数
    label_test = torch.from_numpy(label_test).type(torch.LongTensor)
    ground_xy_train = torch.from_numpy(ground_xy_train).type(torch.LongTensor)
    ground_xy_test = torch.from_numpy(ground_xy_test).type(torch.LongTensor)
    ground_xy_allData = torch.from_numpy(ground_xy_allData).type(torch.LongTensor)
    unlabeled_xy = torch.from_numpy(unlabeled_xy).type(torch.LongTensor)

    print('train：', len(label_train))

    ms4 = to_tensor(ms4_np)
    pan = to_tensor(pan_np)
    pan = np.expand_dims(pan, axis=0) # 给 pan 数组在指定的轴（axis=0）上增加一个新的维度
    #使其形状从 (H, W) 变为 (1, H, W)，表示这是一个包含单张图像的数据集
    ms4 = np.array(ms4).transpose((2, 0, 1)) # ms4 数组的维度从 (H, W, C) 转置为 (C, H, W)

    ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
    pan = torch.from_numpy(pan).type(torch.FloatTensor)

    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    unlabeled_data = MyData1(ms4, pan, unlabeled_xy, Ms4_patch_size)

    FE_ms = model.MS_Model().to(device)
    FE_pan = model.PAN_Model().to(device)

    decoder = model.Decoder().to(device)
    
    MSDP = model.DictionaryRepresentationModule1().to(device)
    PANDP = model.DictionaryRepresentationModule1().to(device)

    optimizer = torch.optim.Adam(list(FE_ms.parameters()) + list(decoder.parameters()) +
                                list(FE_pan.parameters()) + list(MSDP.parameters()) +
                                list(PANDP.parameters()), **config['optimizer']['params'])
    torch.nn.utils.clip_grad_norm_(list(FE_ms.parameters()) + list(decoder.parameters())  +
                                list(FE_pan.parameters())  + list(MSDP.parameters()) +
                                list(PANDP.parameters()), max_norm=1.0)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    #每 5 个 epoch 就进行一次学习率重启，使用余弦退火策略调整学习率

    trainer = Trainer(FE_ms=FE_ms, FE_pan=FE_pan, decoder=decoder, 
                     MSDP=MSDP, PANDP=PANDP, optimizer=optimizer, 
                      scheduler=scheduler,device=device,**config['trainer']) 
                      
    trainer.train(unlabeled_data) # 开始训练，unlabeled_data 是训练的数据集，包含无标签数据


if __name__ == '__main__':
    
    my_main()








