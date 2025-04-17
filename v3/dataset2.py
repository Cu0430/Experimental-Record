import sys 
sys.path.append('/home/aistudio/work/ex')
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import os
from shutil import copyfile 

windowSize = 11 #窗口大小


class MyData(Dataset):
    def __init__(self, MS4, Pan, Label, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan
        self.train_labels = Label
        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

        self.random_flip = transforms.RandomHorizontalFlip()

    def __getitem__(self, index): # 根据索引 index 返回对应的数据样本
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)   
        y_pan = int(4 * y_ms)

        # 裁剪
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]  # dim：chw

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]  

        target = self.train_labels[index]  # 获取当前样本的标签 target

        image_ms_view1 = self.random_flip(image_ms)
        image_ms_view2 = self.random_flip(image_ms)

        image_pan_view1 = self.random_flip(image_pan)
        image_pan_view2 = self.random_flip(image_pan)

        return image_ms_view1, image_ms_view2, image_pan_view1, image_pan_view2, target, locate_xy

    def __len__(self):
        return len(self.gt_xy)

class MyData1(Dataset):
    def __init__(self, MS4, Pan, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan

        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size *4
        self.random_flip = transforms.RandomHorizontalFlip()

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms) 
        y_pan = int(4 * y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        image_ms = self.random_flip(image_ms)
        
        image_pan = self.random_flip(image_pan)
       
        # 目标标签为1
        return image_ms, image_pan, 1, locate_xy

    def __len__(self):
        return len(self.gt_xy)
    
def _create_model_training_folder(writer, files_to_same):
    # 路径为 writer.log_dir 目录下的 checkpoints 子目录
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(model_checkpoints_folder, os.path.basename(file)))