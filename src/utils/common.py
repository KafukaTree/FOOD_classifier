import torch


def true_num(pre_vecs,gt_vecs):
    # pre_vecs: [batch_size, num_classes]
    # gt_vecs: [batch_size, num_classes]
    return torch.sum(pre_vecs.argmax(dim=1) == gt_vecs.argmax(dim=1))

# pred = torch.tensor([[0.1,0.2,0.3,0.4],[0.1,0.2,0.3,0.4],[0.1,0.2,0.8,0.4]])
# gt = torch.tensor([[0,0,0,1],[1,0,0,0],[0,0,1,0]])
# print(true_num(pred,gt)) #应该是1




def label_to_string(label):
    pass
