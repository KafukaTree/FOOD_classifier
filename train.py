#food101训练py文件
import torch
from torch.utils.data import DataLoader
from models import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 接受参数
args = parse_args()
# 加载数据集
train_dataset = FoodDataset(args.data_dir,use_flag=0)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_dataset = FoodDataset(args.data_dir,use_flag=1)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

loss_fn = torch.nn.CrossEntropyLoss()
model = food_can01().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
best_acc = 0
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
        if (i+1)%1000 == 0:
            # 测试模型
            t_num = 0
            for j , (images, gt_vecs) in enumerate(test_loader):
                images = images.to(device)
                gt_vecs = gt_vecs.to(device)
                pred_vecs = model(images)
                t_num = t_num+true_num(pred_vecs, gt_vecs)

            acc = t_num/len(test_dataset)
            if acc > best_acc:
                best_acc = acc
                # 保存最佳参数
                torch.save(model.state_dict(), f"generate/model_pth/model_best.pth")
                print(f"Epoch: {epoch}/{args.epochs}, Batch: {i}/{len(train_loader)}, Acc: {acc}, Best Acc: {best_acc}")
                print('最佳参数已经保存到：', f"generate/model_pth/model_best.pth")
                #保存bestacc
                with open('generate/model_pth/best_acc.txt', 'w') as f:
                    f.write(str(best_acc))
            torch.save(model.state_dict(), f"generate/model_pth/model_last.pth")
            print(f"Epoch: {epoch}/{args.epochs}, Batch: {i}/{len(train_loader)}, Acc: {acc}, Best Acc: {best_acc}")
            print('最新参数已经保存到：', f"generate/model_pth/model_last.pth")

    # 保存模型
    # torch.save(model.state_dict(), f"models/model_epoch_{epoch}.pth")


