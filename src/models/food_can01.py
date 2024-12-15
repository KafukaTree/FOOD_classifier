import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size=3, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def conv_bn(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
        nn.BatchNorm2d(out_channels),
    )


class channel_atten(nn.Module):
    def __init__(self, channel, reduction=16):
        super(channel_atten, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class BLOCK(nn.Module):
    def __init__(self, channel):
        super(BLOCK, self).__init__()
        self.conv = default_conv(channel, channel)
        self.ca = channel_atten(channel)


    def forward(self, x):
        x0 = x
        x = self.conv(x)
        x = self.conv(x)
        x = self.ca(x)
        return x + x0


class RG(nn.Module):
    def __init__(self, channel, input_size, block_num=4):
        super(RG, self).__init__()
        self.block_num = block_num
        self.channel = channel
        self.input_size = input_size

        self.BlOCK = BLOCK(channel)
        self.conv = default_conv(channel, channel)

    def forward(self, x):
        x0 = x
        for i in range(self.block_num):
            x = self.BlOCK(x)
        x = x + x0
        x = self.conv(x)
        return x


class BNTK1(nn.Module):
    def __init__(self, input_size, in_channel, out_channel, stride):
        super(BNTK1, self).__init__()

        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel * 4, kernel_size=1),
            nn.BatchNorm2d(out_channel * 4),
        )
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_channel, out_channel * 4, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channel * 4),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = x
        x = self.conv_bn_relu(x)
        x = x + self.conv_bn(x0)
        x = self.relu(x)
        # print('b1',x.shape)
        return x


class BNTK2(nn.Module):
    def __init__(self, input_size, in_channel):
        super(BNTK2, self).__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 4, kernel_size=1),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // 4, in_channel // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // 4, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = x
        # print(x.shape)
        x = self.conv_bn_relu(x)
        x = x + x0
        x = self.relu(x)
        # print('b2',x.shape)
        return x


class stage(nn.Module):
    def __init__(self, input_size, in_channel, channel_l, BNTK1_num, BNTK2_num, stride=2):
        super(stage, self).__init__()
        self.BNTK1 = BNTK1(input_size, in_channel, channel_l, stride)
        self.BNTK2 = BNTK2(input_size, channel_l*4)
        self.BNTK1_num = BNTK1_num
        self.BNTK2_num = BNTK2_num

    def forward(self, x):
        for i in range(self.BNTK1_num):
            x = self.BNTK1(x)
        for i in range(self.BNTK2_num):
            x = self.BNTK2(x)
        return x

new_layer_replace_ = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    RG(64, 224),
)
class new_layer_replace(nn.Module):
    def __init__(self):
        super(new_layer_replace, self).__init__()
        self.new_layer = nn.Sequential(*new_layer_replace_)
    def forward(self, x):
        x = self.new_layer(x)
        return x
# x = new_layer_replace(224, 3, 101)
# print(x)

class food_can01(nn.Module):
    def __init__(self, input_size=224, input_channel=3, out_channel=101):
        super(food_can01, self).__init__()
        head = [
            nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1),
        ]
        body = [
            RG(64, input_size),
            stage(input_size, 64, 64, BNTK1_num=1, BNTK2_num=2, stride=1),
            stage(input_size, 256, 128, BNTK1_num=1, BNTK2_num=3,stride=2),
            stage(input_size//2, 512, 256, BNTK1_num=1, BNTK2_num=5,stride=2),
            stage(input_size//4, 1024, 512, BNTK1_num=1, BNTK2_num=2,stride=2),
        ]
        tail = [
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, out_channel),
            nn.Softmax(dim=1)
        ]
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x = self.head(x)
        # print(x.shape)
        x = self.body(x)
        # print(x.shape)
        x = self.tail(x)
        return x


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# x = torch.randn(1, 3, 224, 224).to(device)
# model = food_can01(out_channel=22).to(device)
# print(model(x).shape)
