
import model2
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from tifffile import imread
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from tqdm import tqdm
import os
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

os.environ["OMP_NUM_THREADS"] = "1"
device = torch.device('cuda:0')
# your pre-training result
pretrained_folder = 'Apr17_17-08-11_autodl-container-bcde4980ca-473851e8'

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.FE_ms=model2.MS_Model()
        self.FE_pan=model2.PAN_Model()
        self.MSDP=model2.DictionaryRepresentationModule_m()
        self.PANDP=model2.DictionaryRepresentationModule_p()
        
        self.Fusion=model2.ModalityAwareFPN()
        with torch.no_grad():
            dummy_ms = torch.randn(1, 512, 8, 8)
            dummy_ms1 = torch.randn(1, 1024, 8, 8)
            dummy_ms2 = torch.randn(1, 2048, 8, 8)
            dummy_pan = torch.randn(1, 512, 8, 8)
            dummy_pan1 = torch.randn(1, 1024, 8, 8)
            dummy_pan2 = torch.randn(1, 2048, 8, 8)
            self.Fusion(dummy_ms, dummy_ms1, dummy_ms2, dummy_pan, dummy_pan1, dummy_pan2)
        
        if pretrained_folder:
            try:
                checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')
                load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                                         map_location=torch.device(torch.device(device)))
                
                self.FE_ms.load_state_dict(load_params['FE_ms_state_dict'])
                self.FE_pan.load_state_dict(load_params['FE_pan_state_dict'])
                self.MSDP.load_state_dict(load_params['MSDP_state_dict'])
                self.PANDP.load_state_dict(load_params['PANDP_state_dict'])
                self.Fusion.load_state_dict(load_params['Fusion_state_dict'])                             
                print("the model of MS&PAN has been loaded")
            except FileNotFoundError:
                print("Pre-trained weights not found. Training from scratch.")
        
        self.add_block = nn.Sequential(
            #nn.Linear(256,128),
            #nn.BatchNorm1d(128),
            #nn.ReLU(True),
            #nn.Dropout(p=0.5),
            nn.Linear(256,64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)  # 最终输出类别数
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 8 * 8, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, ms, pan):
        
        ms2=self.FE_ms.forward_stem(ms) #[128,8,16,16]
        pan2=self.FE_pan.forward_stem(pan) #[128,8,64,64]
        
        MSDP_ms,_=self.MSDP(ms2) 
        PANDP_pan,_=self.PANDP(pan2)
        
        _,_,_,ms3_1,ms4_1,ms5_1=self.FE_ms(MSDP_ms)
        _,_,_,pan3_1,pan4_1,pan5_1=self.FE_pan(PANDP_pan)
        P3, P4, P5 = self.Fusion(ms3_1, ms4_1, ms5_1, pan3_1, pan4_1, pan5_1)
        
        P4_up = nn.functional.interpolate(P4, size=(8, 8), mode='bilinear', align_corners=False)
        P5_up = nn.functional.interpolate(P5, size=(8, 8), mode='bilinear', align_corners=False)

        # 通道拼接融合
        fused = torch.cat([P3, P4_up, P5_up], dim=1)  # [128, 256*3, 8, 8]
        fused = fused.view(fused.size(0), -1)         # 展平为 [128, 256*3*8*8]

        # 分类预测
        out = self.classifier(fused)
        '''
        _,_,ms_f,_,_,_=self.FE_ms(MSDP_ms)
                
        _,_,pan_f,_,_,_=self.FE_pan(PANDP_pan)
        
        feature = torch.cat([ms_f, pan_f], 1)
        feature = feature.view(feature.size()[0], -1) 
        out = self.add_block(feature)
        '''
        return out


Train_Rate = 0.01
BATCH_SIZE = 128
EPOCH = 30

pan_np = imread('/root/autodl-tmp/datasets/hhht/pan.tif')
print('The shape of the original PAN:', np.shape(pan_np))

ms4_np = imread('/root/autodl-tmp/datasets/hhht/ms4.tif')
print('The shape of the original MS:', np.shape(ms4_np))

label_np = np.load("/root/autodl-tmp/datasets/hhht/train.npy")
print('The shape of the train label', np.shape(label_np))

test_label_np = np.load("/root/autodl-tmp/datasets/hhht/test.npy")
print('The shape of the test label', np.shape(test_label_np ))


Ms4_patch_size = 16
Interpolation = cv2.BORDER_REFLECT_101
top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2),  # 7  8
                                                int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2))  # 7  8
ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)  # 长宽各扩15
print('The shape of the MS after padding:', np.shape(ms4_np))

Pan_patch_size = Ms4_patch_size *4
top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2),  # 28 32
                                                int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2))  # 28 32
pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)  # 长宽各扩60
print('The shape of the PAN after padding:', np.shape(pan_np))

# label_np=label_np.astype(np.uint8)
label_np = label_np - 1

label_element, element_count = np.unique(label_np, return_counts=True)
print('Class label:', label_element)
print('Number of samples in each category:', element_count)
Categories_Number = len(label_element) - 1
print('Number of categories labeled:', Categories_Number)

test_label_np = test_label_np - 1

test_label_element, test_element_count = np.unique(test_label_np, return_counts=True)
print('Test class label:', test_label_element)
print('Number of test samples in each category:', test_element_count)
test_Categories_Number = len(test_label_element) - 1
print('Number of test categories labeled:', test_Categories_Number)

def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image
#统计 训练label 图中的 row、column 所有标签个数,并统计每一类的个数
label_row, label_column = np.shape(label_np)
ground_xy = np.array([[]] * Categories_Number).tolist()
ground_xy_allData = np.arange(label_row * label_column * 2).reshape(label_row * label_column, 2)
count = 0
for row in range(label_row):
    for column in range(label_column):
        ground_xy_allData[count] = [row, column]
        count = count + 1
        if label_np[row][column] != 255:
            ground_xy[int(label_np[row][column])].append([row, column])

#统计 测试label 图中的 row、column 所有标签个数,并统计每一类的个数
test_label_row, test_label_column = np.shape(test_label_np)
test_ground_xy = np.array([[]] * test_Categories_Number).tolist()
test_ground_xy_allData = np.arange(test_label_row * test_label_column * 2).reshape(test_label_row * test_label_column, 2)
test_count = 0
for test_row in range(test_label_row):
    for test_column in range(test_label_column):
        test_ground_xy_allData[test_count] = [test_row, test_column]
        test_count = test_count + 1
        if test_label_np[test_row][test_column] != 255:
            test_ground_xy[int(test_label_np[test_row][test_column])].append([test_row, test_column])

#标签内打乱
for categories in range(Categories_Number):
    ground_xy[categories] = np.array(ground_xy[categories])
    shuffle_array = np.arange(0, len(ground_xy[categories]), 1)
    np.random.shuffle(shuffle_array)
    ground_xy[categories] = ground_xy[categories][shuffle_array]

shuffle_array = np.arange(0, label_row * label_column, 1)
np.random.shuffle(shuffle_array)
ground_xy_allData = ground_xy_allData[shuffle_array]

#测试标签内打乱
for test_categories in range(test_Categories_Number):  
    test_ground_xy[test_categories] = np.array(test_ground_xy[test_categories])
    test_shuffle_array = np.arange(0, len(test_ground_xy[test_categories]), 1)  
    np.random.shuffle(test_shuffle_array)
    test_ground_xy[test_categories] = test_ground_xy[test_categories][test_shuffle_array]  

test_shuffle_array = np.arange(0, test_label_row * test_label_column, 1)
np.random.shuffle(test_shuffle_array)
test_ground_xy_allData = test_ground_xy_allData[test_shuffle_array]

ground_xy_train = []
ground_xy_test = []
label_train = []
label_test = []

for categories in range(Categories_Number):
    categories_number = len(ground_xy[categories])
    for i in range(categories_number):
        if i < int(categories_number * Train_Rate):
            ground_xy_train.append(ground_xy[categories][i])
    label_train = label_train + [categories for x in range(int(categories_number * Train_Rate))]
    #label_train = label_train + [categories for x in range(int(categories_number))]

for test_categories in range(test_Categories_Number):
    test_categories_number = len(test_ground_xy[test_categories])
    for i in range(test_categories_number):
        ground_xy_test.append(test_ground_xy[test_categories][i])
    label_test = label_test + [test_categories for x in range(int(test_categories_number))]

label_train = np.array(label_train)
label_test = np.array(label_test)
ground_xy_train = np.array(ground_xy_train)
ground_xy_test = np.array(ground_xy_test)

shuffle_array = np.arange(0, len(label_test), 1)
np.random.shuffle(shuffle_array)
label_test = label_test[shuffle_array]
ground_xy_test = ground_xy_test[shuffle_array]

shuffle_array = np.arange(0, len(label_train), 1)
np.random.shuffle(shuffle_array)
label_train = label_train[shuffle_array]
ground_xy_train = ground_xy_train[shuffle_array]

label_train = torch.from_numpy(label_train).type(torch.LongTensor)
label_test = torch.from_numpy(label_test).type(torch.LongTensor)
ground_xy_train = torch.from_numpy(ground_xy_train).type(torch.LongTensor)
ground_xy_test = torch.from_numpy(ground_xy_test).type(torch.LongTensor)
ground_xy_allData = torch.from_numpy(ground_xy_allData).type(torch.LongTensor)

print('Number of training samples:', len(label_train))
print('Number of test samples:', len(label_test))


ms4 = to_tensor(ms4_np)
pan = to_tensor(pan_np)
pan = np.expand_dims(pan, axis=0)
ms4 = np.array(ms4).transpose((2, 0, 1))

ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
pan = torch.from_numpy(pan).type(torch.FloatTensor)


class MyData(Dataset):
    def __init__(self, MS4, Pan, Label, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan
        self.train_labels = Label
        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size*4
        #self.transform=transform

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)
        y_pan = int(4 * y_ms)

        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]  # dim：chw

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]
        #if self.transform:
         #   image_ms = self.transform(image_ms)
          #  image_pan = self.transform(image_pan)
        locate_xy = self.gt_xy[index]

        target = self.train_labels[index]

        return image_ms, image_pan, target, locate_xy

    def __len__(self):
        return len(self.gt_xy)


class MyData1(Dataset):
    def __init__(self, MS4, Pan, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan

        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size*4

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)
        y_pan = int(4 * y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]
        
        locate_xy = self.gt_xy[index]

        return image_ms, image_pan, locate_xy

    def __len__(self):
        return len(self.gt_xy)

train_data = MyData(ms4, pan, label_train, ground_xy_train, Ms4_patch_size)
test_data = MyData(ms4, pan, label_test, ground_xy_test, Ms4_patch_size)
all_data = MyData1(ms4, pan, ground_xy_allData, Ms4_patch_size)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
all_data_loader = DataLoader(dataset=all_data, batch_size=BATCH_SIZE, shuffle=False,num_workers=0)

model = Net(num_classes=Categories_Number).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0007, weight_decay=1e-4) #1e-3
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)


def train_model(model, train_loader, optimizer, epoch):
    model.train()
    correct = 0.0
    iters=len(train_loader)*epoch
    #iters = epoch * len(label_train) // BATCH_SIZE

    for step, (ms, pan, label, _) in enumerate(train_loader):
        ms, pan, label = ms.to(device), pan.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(ms, pan)
        pred_train = output.max(1, keepdim=True)[1]
        correct += pred_train.eq(label.view_as(pred_train).long()).sum().item()
        loss = F.cross_entropy(output, label.long())
        loss.backward()
        optimizer.step()
        scheduler.step(epoch + step / iters)
        if step % 20 == 0:
            print("Train Epoch: {} \t Loss : {:.6f} \t step: {} ".format(epoch, loss.item(), step))
    print("Train Accuracy: {:.6f}".format(correct * 100.0 / len(train_loader.dataset)))


def test_third(model, test_loader):
    model.eval()
    correct = 0.0
    test_bar=tqdm(test_loader)
    test_metric=np.zeros([test_Categories_Number,test_Categories_Number])
    with torch.no_grad():
        for data, data1, target, _  in test_bar:
            data, data1, target= data.to(device), data1.to(device), target.to(device)
            output= model(data,data1)
            # test_loss = F.cross_entropy(output[0], target.long()).item()
            pred = output.max(1, keepdim=True)[1]
            for i in range(len(target)):
                test_metric[int(pred[i].item())][int(target[i].item())]+=1
            correct += pred.eq(target.view_as(pred).long()).sum().item()
        print("test Accuracy:{:.3f} \n".format( 100.0 * correct / len(test_loader.dataset)))
    b=np.sum(test_metric,axis=0)
    accuracy=[]
    c=0
    for i in range(0,test_Categories_Number):
        a=test_metric[i][i]/b[i]
        accuracy.append(a)
        c+=test_metric[i][i]
        print('category {0:d}: {1:f}'.format(i,a))
    average_accuracy = np.mean(accuracy)
    overall_accuracy = c/np.sum(b,axis=0)
    kappa_coefficient = kappa(test_metric)
    print('AA: {0:f}'.format(average_accuracy))
    print('OA: {0:f}'.format(overall_accuracy))
    print('KAPPA: {0:f}'.format(kappa_coefficient))
    return 100.0 * correct / len(test_loader.dataset)


def aa_oa(matrix):
    accuracy = []
    b = np.sum(matrix, axis=0)
    c = 0
    on_display = []
    for i in range(1, matrix.shape[0]):
        a = matrix[i][i]/b[i]
        c += matrix[i][i]
        accuracy.append(a)
        on_display.append([b[i], matrix[i][i], a])
        print("Category:{}. Overall:{}. Correct:{}. Accuracy:{:.6f}".format(i, b[i], matrix[i][i], a))
    aa = np.mean(accuracy)
    oa = c / np.sum(b, axis=0)
    k = kappa(matrix)
    print("OA:{:.6f} AA:{:.6f} Kappa:{:.6f}".format(oa, aa, k))
    return [aa, oa, k, on_display]

def kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)
    return (po - pe) / (1 - pe)

import time
start=time.time()
for epoch in range(1, EPOCH+1):
    train_model(model,  train_loader, optimizer, epoch)
test_matrix = test_third(model, test_loader)
print(test_matrix)
'''
torch.save(model, './transfer_model.pkl')
cnn = torch.load('./transfer_model.pkl')
cnn.to(device)

class_count = np.zeros(Categories_Number)
out_clour = np.zeros((label_row, label_column, 3))
gt_clour = np.zeros((label_row, label_column, 3))
def clour_model(cnn, all_data_loader):
    train_bar = tqdm(all_data_loader)
    for step, (ms4, pan, gt_xy) in enumerate(train_bar):
        ms4 = ms4.to(device)
        pan = pan.to(device)
        with torch.no_grad():
            output = cnn(ms4, pan)
        pred_y = torch.max(output, 1)[1].to(device).data.squeeze()
        pred_y_numpy = pred_y.cpu().numpy()
        #print("pred_y###",pred_y_numpy)
        gt_xy = gt_xy.numpy()
        for k in range(len(gt_xy)):
            if pred_y_numpy[k] == 0:
                class_count[0] = class_count[0] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [255, 255, 0]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [255, 255, 0]
            elif pred_y_numpy[k] == 1:
                class_count[1] = class_count[1] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [255, 0, 0]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [255, 0, 0]
            elif pred_y_numpy[k] == 2:
                class_count[2] = class_count[2] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [33, 145, 237]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [33, 145, 237]
            elif pred_y_numpy[k] == 3:
                class_count[3] = class_count[3] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [201, 252, 189]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [201, 252, 189]
            elif pred_y_numpy[k] == 4:
                class_count[4] = class_count[4] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [0, 0, 230]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [0, 0, 230]
            elif pred_y_numpy[k] == 5:
                class_count[5] = class_count[5] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [0, 255, 0]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [0, 255, 0]
            elif pred_y_numpy[k] == 6:
                class_count[6] = class_count[6] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [240, 32, 160]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [240, 32, 160]
            elif pred_y_numpy[k] == 7:
                class_count[7] = class_count[7] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [221, 160, 221]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [221, 160, 221]
            elif pred_y_numpy[k] == 8:
                class_count[7] = class_count[8] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [140, 230, 240]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [140, 230, 240]
            elif pred_y_numpy[k] == 9:
                class_count[7] = class_count[9] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [0, 255, 255]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [0, 255, 255]
            if label_np[gt_xy[k][0]][gt_xy[k][1]] == 255:
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [0, 0, 0]
    print(class_count)
    cv2.imwrite("/home/gpu/Experiment/gpt/exp/clour_images/hhht.png", out_clour)
    cv2.imwrite("/home/gpu/Experiment/gpt/exp/clour_images/hhht_gt.png", gt_clour)

clour_model(model,  all_data_loader)


end=time.time()
print("The running time of the program is：{}".format(end-start))


print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
'''