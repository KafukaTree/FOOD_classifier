# food101的数据集loader
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np
import PIL
import random
import torchvision
#读取txt的字符串并组成字符串列表，换行为分界
def read_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]
# txt = read_txt('D:/food/dataset/food-101/meta/classes.txt')
# print(txt)
Transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 先调整大小
    transforms.RandomRotation(30),  # 然后随机旋转
    transforms.RandomHorizontalFlip(),  # 接着随机水平翻转
    transforms.ToTensor(),  # 再转换为张量
    #颜色扰动
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 最后标准化
])
class FoodDataset(Dataset):
    def __init__(self, root_dir, use_flag = 0,transform=Transform):
        self.root_dir = root_dir
        self.transform = transform
        self.use_flag = use_flag #0是train,1是test
        self.images_dir = root_dir + '/images'
        self.meta_dir = root_dir + '/meta'
        self.classes_txt = self.meta_dir +'/classes.txt'
        self.labels_txt = self.meta_dir + '/labels.txt'
        self.train_txt = self.meta_dir + '/train.txt'
        self.test_txt = self.meta_dir + '/test.txt'
        self.train_index = read_txt(self.train_txt)
        self.test_index = read_txt(self.test_txt)
        self.labels = read_txt(self.labels_txt)


        self.train_len = len(self.train_index)
        self.test_len = len(self.test_index)
        self.classes = read_txt(self.classes_txt)

    def __len__(self):
        if self.use_flag == 0:
            return self.train_len
        else:
            return self.test_len
    def __getitem__(self, idx):
        if self.use_flag == 0:
            image_pth = self.images_dir+'/'+ self.train_index[idx] +'.jpg'
            gt_vec = np.zeros(len(self.classes))
            gt_vec[self.classes.index(self.train_index[idx].split('/')[0])] = 1

        else:
            image_pth = self.images_dir + '/' + self.test_index[idx] +'.jpg'
            gt_vec = np.zeros(len(self.classes))
            gt_vec[self.classes.index(self.test_index[idx].split('/')[0])] = 1


        image = Image.open(image_pth).convert('RGB')
        if self.use_flag ==0 and self.transform:
                image = self.transform(image)
                # 保存image,tensor类型,用torchvision
                #先创建文件夹

                torchvision.utils.save_image(image, 'D:/food/src/generate/temp/' + self.train_index[idx] + '.jpg')

                # 保存label，gt,tensor类型
                torch.save(torch.tensor(gt_vec), 'D:/food/src/generate/temp/' + self.train_index[idx] + '.pt')

        elif self.use_flag ==1:
            #resize加归一化
            image = transforms.Resize((224, 224))(image)
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)




        return image, torch.tensor(gt_vec)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = FoodDataset(root_dir='D:/food/dataset/food-101', use_flag=0,transform=Transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
for i,(x,y) in enumerate(train_loader):
    print(x.shape,y.shape)
    if i>=5:
        break

