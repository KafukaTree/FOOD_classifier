# 接受命令行参数的函数
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_dir', type=str, default='../dataset/food-101', help='data path')
    parser.add_argument('--save_path', type=str, default='save', help='save path')
    parser.add_argument('--model_path', type=str, default='model', help='model path')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='num workers')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')

    args = parser.parse_args()
    return args