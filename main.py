# -*- coding: utf-8 -*- 
# @Time : 2021/3/30 15:25 
# @Author : CHENTian
# @File : main.py

import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
import torch.optim as optim
from MyDataset import *
import datetime

class SE_VGG(nn.Module):
    def __init__(self, num_classes):
        super(SE_VGG, self).__init__()
        self.num_classes = num_classes

        # block1
        self.block1_conv1 = nn.Conv2d(in_channels=3, out_channels=64, padding=1, kernel_size=3, stride=1)
        self.block1_conv2 = nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=3, stride=1)
        self.block1_bn = nn.BatchNorm2d(64)

        # block2
        self.block2_conv1 = nn.Conv2d(in_channels=64, out_channels=128, padding=1, kernel_size=3, stride=1)
        self.block2_conv2 = nn.Conv2d(in_channels=128, out_channels=128, padding=1, kernel_size=3, stride=1)
        self.block2_bn = nn.BatchNorm2d(128)

        # block3
        self.block3_conv1 = nn.Conv2d(in_channels=128, out_channels=256, padding=1, kernel_size=3, stride=1)
        self.block3_conv2 = nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3, stride=1)
        self.block3_conv3 = nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3, stride=1)
        self.block3_bn = nn.BatchNorm2d(256)

        # block4
        self.block4_conv1 = nn.Conv2d(in_channels=256, out_channels=512, padding=1, kernel_size=3, stride=1)
        self.block4_conv2 = nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1)
        self.block4_conv3 = nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1)
        self.block4_bn = nn.BatchNorm2d(512)

        # block5
        self.block5_conv1 = nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1)
        self.block5_conv2 = nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1)
        self.block5_conv3 = nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1)
        self.block5_bn = nn.BatchNorm2d(512)

        # block6
        self.block6_fc1 =  nn.Linear(in_features=512*7*7, out_features=4096)
        self.block6_fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.block6_softmax = nn.Linear(in_features=4096, out_features=self.num_classes)

        # common func
        self.common_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.common_dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # block1
        block1_conv1_out = F.relu(self.block1_conv1(x))
        block1_conv1_out = self.block1_bn(block1_conv1_out)
        block1_conv2_out = F.relu(self.block1_conv2(block1_conv1_out))
        block1_conv2_out = self.block1_bn(block1_conv2_out)
        block1_pool_out = self.common_pool(block1_conv2_out)

        # block2
        block2_conv1_out = F.relu(self.block2_conv1(block1_pool_out))
        block2_conv1_out = self.block2_bn(block2_conv1_out)
        block2_conv2_out = F.relu(self.block2_conv2(block2_conv1_out))
        block2_conv2_out = self.block2_bn(block2_conv2_out)
        block2_pool_out = self.common_pool(block2_conv2_out)

        # block3
        block3_conv1_out = F.relu(self.block3_conv1(block2_pool_out))
        block3_conv1_out = self.block3_bn(block3_conv1_out)
        block3_conv2_out = F.relu(self.block3_conv2(block3_conv1_out))
        block3_conv2_out = self.block3_bn(block3_conv2_out)
        block3_conv3_out = F.relu(self.block3_conv3(block3_conv2_out))
        block3_conv3_out = self.block3_bn(block3_conv3_out)
        block3_pool_out = self.common_pool(block3_conv3_out)

        # block4
        block4_conv1_out = F.relu(self.block4_conv1(block3_pool_out))
        block4_conv1_out = self.block4_bn(block4_conv1_out)
        block4_conv2_out = F.relu(self.block4_conv2(block4_conv1_out))
        block4_conv2_out = self.block4_bn(block4_conv2_out)
        block4_conv3_out = F.relu(self.block4_conv3(block4_conv2_out))
        block4_conv3_out = self.block4_bn(block4_conv3_out)
        block4_pool_out = self.common_pool(block4_conv3_out)

        # block5
        block5_conv1_out = F.relu(self.block5_conv1(block4_pool_out))
        block5_conv1_out = self.block5_bn(block5_conv1_out)
        block5_conv2_out = F.relu(self.block5_conv2(block5_conv1_out))
        block5_conv2_out = self.block5_bn(block5_conv2_out)
        block5_conv3_out = F.relu(self.block5_conv3(block5_conv2_out))
        block5_conv3_out = self.block5_bn(block5_conv3_out)
        block5_pool_out = self.common_pool(block5_conv3_out)
        block5_pool_out = block5_pool_out.view(-1, 512*7*7)
        # block6

        block6_fc1_out = F.relu(self.block6_fc1(block5_pool_out))
        block6_dropout1 = self.common_dropout(block6_fc1_out)
        block6_fc2_out = F.relu(self.block6_fc2(block6_dropout1))
        block6_dropout2 = self.common_dropout(block6_fc2_out)
        block6_softmax_out = self.block6_softmax(block6_dropout2)

        return block6_softmax_out


if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #train_set = MyDataset("./data/", "train.txt", transform)
    train_set = datasets.ImageFolder(root="./data/coco_animals/train", transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=20, shuffle=True, num_workers=2)
    test_set = datasets.ImageFolder(root="./data/coco_animals/val", transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=False)
    """
    print(torch.cuda.get_device_name())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg = SE_VGG(num_classes=8)
    vgg.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg.parameters(),lr=0.0001, momentum=0.9)
    start = datetime.datetime.now()
    for epoch in range(500):
        running_loss = 0.0
        running_acc = 0.0
        for i , (image, label) in enumerate(train_loader):
            # 训练模式
            vgg.train()
            image, label = image.to(device), label.to(device)
            outputs = vgg(image)
            loss = criterion(outputs, label)

            running_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            num_correct = (pred == label).float().mean()
            running_acc += num_correct.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, running_loss / (len(train_set)), running_acc / (len(train_set))))

        vgg.eval()
        eval_loss = 0
        eval_acc = 0
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            outputs = vgg(image)
            loss = criterion(outputs, label)
            eval_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            num_correct = (pred == label).sum()
            eval_acc += num_correct.item()
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
            test_set)), eval_acc / (len(test_set))))
        print()
    end = datetime.datetime.now()
    print((end - start))
    PATH = './vgg16_net1.pth'
    torch.save(vgg.state_dict(), PATH)
    print("Finished Training")

