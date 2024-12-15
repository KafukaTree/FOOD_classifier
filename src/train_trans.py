import torch
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn as nn
from utils import *
from models import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 接受参数
args = parse_args()
# 加载预训练的 ResNet50 模型
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# 打印原始模型结构，了解需要修改的部分
# print(resnet50)

# 创建一个新的卷积层作为替代（这里只是一个示例）
new_conv_layer = new_layer_replace()
# 替换第一层卷积层（相当于移除了 "stage0" 的 Conv 和 MaxPool）
resnet50.conv1 = new_conv_layer
resnet50.fc = nn.Linear(2048, 101)  # 修改全连接层以适应新的输出维度
# 导入参数
resnet50.load_state_dict(torch.load('generate/model_pth/model_last.pth'))
# for name, child in resnet50.named_children():
#     if name in ['layer1', 'layer2', 'layer3']:
#         for param in child.parameters():
#             param.requires_grad = False
# resnet50.bn1 = None
# resnet50.relu = None
# resnet50.maxpool = None


# 加载数据集
train_dataset = FoodDataset(args.data_dir,use_flag=0)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_dataset = FoodDataset(args.data_dir,use_flag=1)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=args.num_workers)

loss_fn = torch.nn.CrossEntropyLoss()
model = resnet50.to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, resnet50.parameters()), lr=args.lr, weight_decay=args.weight_decay)
best_acc = 0
# 读取 best_acc.txt
with open('generate/model_pth/best_acc.txt', 'r') as f:
    best_acc = float(f.read())
print('best_acc:',best_acc)
# 训练循环
for epoch in range(args.epochs):
    # 训练模型
    for i , (images, gt_vecs) in enumerate(train_loader):
        images = images.to(device)
        gt_vecs = gt_vecs.to(device)
        pred_vecs = model(images)

        # 求多个batch的loss
        loss = torch.zeros(len(gt_vecs)).to(device)
        for k in range(len(gt_vecs)):
            loss[k] = loss_fn(pred_vecs[k], gt_vecs[k])
        loss = torch.mean(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i)%20 == 0:
            print(f"Epoch: {epoch}/{args.epochs}, Batch: {i}/{len(train_loader)}, Loss: {loss.item()}")
        if (i)%2000 == 0:
            # 测试模型
            t_num = 0
            num = 0
            for j , (images, gt_vecs) in enumerate(test_loader):
                images = images.to(device)
                gt_vecs = gt_vecs.to(device)
                pred_vecs = model(images)
                t_num = t_num+true_num(pred_vecs, gt_vecs)
                num += len(gt_vecs)
                if num >=2000:
                    break

            acc = t_num/num
            print('---------------------------------acc_new:',acc.item(),'---------------------------------------')
            if acc > best_acc:
                best_acc = acc
                # 保存最佳参数
                torch.save(model.state_dict(), f"generate/model_pth/model_best.pth")
                print(f"Epoch: {epoch}/{args.epochs}, Batch: {i}/{len(train_loader)}, Acc: {acc}, Best Acc: {best_acc}")
                print('最佳参数已经保存到：', f"generate/model_pth/model_best.pth")
                #保存bestacc
                with open('generate/model_pth/best_acc.txt', 'w') as f:
                    f.write(str(best_acc.item()))
        if (i)%400 == 0:
            torch.save(model.state_dict(), f"generate/model_pth/model_last.pth")
            print(f"Epoch: {epoch}/{args.epochs}, Batch: {i}/{len(train_loader)}, Best Acc: {best_acc}")
            print('最新参数已经保存到：', f"generate/model_pth/model_last.pth")


# 确认模型结构是否符合预期
# print(resnet50)
# x = (8, 3, 224, 224)
#   # 假设输入图像大小为 224x224，批次大小为 8
# print(resnet50(torch.randn(x)).shape)