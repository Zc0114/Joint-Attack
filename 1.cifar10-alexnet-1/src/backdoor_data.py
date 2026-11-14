#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm
import copy
import torchvision.utils as vutils
import numpy as np
# import cv2
from torch import nn
from torch import autograd

# from utils.options import args_parser
# args = args_parser()
# args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

import matplotlib.pyplot as plt
#忽略警告
import warnings
warnings.filterwarnings('ignore')

###################################################################
def weighted_part_average(name1, name2, name3, p1=0.5, p2=0.5):
    # original image
    # image1 = scipy.misc.imread(name1)
    image1 = name1
    # filter image
    # image2 = scipy.misc.imread(name2)
    image2 = name2
    # print("image1.shape", image1.shape)  #torch.Size([1, 28, 28])
    # print("image2.shape", image2.shape)  #torch.Size([1, 8, 8])
    # image3 = np.copy(image1)
    image3 = image1
    w = image2.shape[1]
    h = image2.shape[2]
    # print(p1, p2) #0,1
    for y in range(h):
        for x in range(w):
            # print("y,x",y,x)
            # if mask[y][x] == 1:
                image3[:,y,x] = p1*image1[:,y,x] + p2*image2[:,y,x]
    # print("image3",image3.shape) #torch.Size([1, 28, 28])
    # scipy.misc.imsave(name3, image3)
    return image3


def filter2(fname1, fname2):
    p1 = float(0)
    p2 = 1 - p1

    image3=weighted_part_average(fname1, fname2, fname1, p1, p2)

    return image3


###################################################################
class MyDataset_cifar_new(Dataset):

    def __init__(self,adv_patch_cpu, dataset, target, portion=0.5, mode="train", device='cpu',number_of_classes=10):
        self.dataset = self.addTrigger(adv_patch_cpu,dataset, target, portion, mode)
        self.device = device
        self.number_of_classes=number_of_classes

    def __getitem__(self, item):
        img = self.dataset[item][0]

        # img1 = img.transpose(1, 2, 0)
        # plt.figure(1)
        # plt.imshow(img1)
        # plt.show()
        img = torch.Tensor(img)

        label = np.zeros(self.number_of_classes)
        label[self.dataset[item][1]] = 1
        label = torch.Tensor(label)
        label = torch.argmax(label, -1)
        img = img.to(self.device)
        label = label.to(self.device)
        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, adv_patch_cpu, dataset, target, portion, mode):
        print("Generating " + mode + " Bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * portion)]
        # print("len(dataset),portion",len(dataset),portion)
        # print("perm ",perm )

        dataset_ = list()
        cnt = 0
        for i in range(len(dataset)):
            data = dataset[i]
            img = np.array(data[0])
            # print(np.max(img))
            img = np.resize(img, (3, 32, 32))

            channels = img.shape[0]
            width = img.shape[1]
            height = img.shape[2]
            pixel_value = 2.82
            distance = 3

            if i in perm:
                # for c in range(channels):
                #     img[c, width - distance, height - distance] = pixel_value
                #     img[c, width - distance - 1, height - distance - 1] = pixel_value
                #     img[c, width - distance, height - distance - 2] = pixel_value
                #     img[c, width - distance - 2, height - distance] = pixel_value

                img = torch.Tensor(img)
                adv_patch_cpu = torch.Tensor(adv_patch_cpu)
                # print("img",img.shape)  #  torch.Size([3, 32, 32])
                # print("adv_patch_cpu",adv_patch_cpu.shape) # torch.Size([3, 3, 3])

                p_img_batch = filter2(img, adv_patch_cpu)
                # print("p_img_batch",p_img_batch.shape) #torch.Size([3, 32, 32])
                img = p_img_batch.data.cpu().numpy()
                # print("img", img.shape) # (3, 32, 32)
                # plt.imshow(img.squeeze(0))
                # plt.show()

                dataset_.append((img, target))
                cnt += 1
            else:
                dataset_.append((img, data[1]))

        print("Injecting Over: " + str(cnt) + " Bad Imgs, " + str(len(dataset) - cnt) + " Clean Imgs") #Injecting Over: 10000 Bad Imgs, 0 Clean Imgs
        return dataset_

class MyDataset_mnist(Dataset):

    def __init__(self,adv_patch_cpu, dataset, target, portion=0.5, mode="train", device='cpu'):
        self.dataset = self.addTrigger(adv_patch_cpu,dataset, target, portion, mode)
        self.device = device

    def __getitem__(self, item):
        img = self.dataset[item][0]

        # img1 = img.transpose(1, 2, 0)
        # plt.figure(1)
        # plt.imshow(img1)
        # plt.show()
        img = torch.Tensor(img)

        label = np.zeros(10)
        label[self.dataset[item][1]] = 1
        label = torch.Tensor(label)
        label = torch.argmax(label, -1)
        img = img.to(self.device)
        label = label.to(self.device)
        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, adv_patch_cpu,dataset, target, portion, mode):
        print("Generating " + mode + " Bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * portion)]
        
        dataset_ = list()
        cnt = 0

        for i in range(len(dataset)):
            data = dataset[i]
            img = np.array(data[0])
            # print(np.max(img))
            img = np.resize(img, (1, 28, 28))
            channels = img.shape[0]
            width = img.shape[1]
            height = img.shape[2]
            pixel_value = 2.82
            distance = 3
            if i in perm:


                # for c in range(channels):
                #     img[c, width - distance, height - distance] = pixel_value
                #     img[c, width - distance - 1, height - distance - 1] = pixel_value
                #     img[c, width - distance, height - distance - 2] = pixel_value
                #     img[c, width - distance - 2, height - distance] = pixel_value

                img = torch.Tensor(img)
                adv_patch_cpu = torch.Tensor(adv_patch_cpu)
                # print("img",img.shape)  #  torch.Size([3, 32, 32])
                # print("adv_patch_cpu",adv_patch_cpu.shape) # torch.Size([3, 3, 3])

                p_img_batch = filter2(img, adv_patch_cpu)
                # print("p_img_batch",p_img_batch.shape) #torch.Size([3, 32, 32])
                img = p_img_batch.data.cpu().numpy()
                # print("img", img.shape) # (3, 32, 32)
                # plt.imshow(img.squeeze(0))
                # plt.show()

                dataset_.append((img, target))
                cnt += 1



            else:
                dataset_.append((img, data[1]))

        print("Injecting Over: " + str(cnt) + " Bad Imgs, " + str(len(dataset) - cnt) + " Clean Imgs")
        return dataset_

class MyDataset_cifar(Dataset):
    def __init__(self, dataset, target, portion=0.5, mode="train", device='cpu'):
        self.dataset = self.addTrigger(dataset, target, portion, mode)
        self.device = device

    def __getitem__(self, item):
        img = self.dataset[item][0]

        # img1 = img.transpose(1, 2, 0)
        # plt.figure(1)
        # plt.imshow(img1)
        # plt.show()

        # img = img[..., np.newaxis]
        #img = np.expand_dims(img, axis=0)
        # img = torch.Tensor(img).permute(2,0,1)
        img = torch.Tensor(img)

        label = np.zeros(10)
        label[self.dataset[item][1]] = 1
        label = torch.Tensor(label)
        label = torch.argmax(label, -1)
        img = img.to(self.device)
        label = label.to(self.device)
        return img, label

    def __len__(self):
        return len(self.dataset)


    def addTrigger(self, dataset, target, portion, mode):
        print("Generating " + mode + " Bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * portion)]
        dataset_ = list()
        cnt = 0
        for i in range(len(dataset)):
            data = dataset[i]
            img = np.array(data[0])
            # print(np.max(img))
            img = np.resize(img, (3, 32, 32))
            channels = img.shape[0]
            width = img.shape[1]
            height = img.shape[2]
            pixel_value = 1
            distance = 3
            if i in perm:
                for c in range(channels):
                    img[c, width - distance, height - distance] = pixel_value
                    img[c, width - distance - 1, height - distance - 1] = pixel_value
                    img[c, width - distance, height - distance - 2] = pixel_value
                    img[c, width - distance - 2, height - distance] = pixel_value

                dataset_.append((img, target))
                cnt += 1
            else:
                dataset_.append((img, data[1]))

        print("Injecting Over: " + str(cnt) + " Bad Imgs, " + str(len(dataset) - cnt) + " Clean Imgs")
        return dataset_

class MyDataset_mnist_DBA(Dataset):

    def __init__(self, args, dataset, target, portion=0.5, epoch=1, mode="train", idx=2, device='cpu'):
        self.dataset = self.addTrigger(args, dataset, target, portion, mode, idx, epoch)
        self.device = device

    def __getitem__(self, item):
        img = self.dataset[item][0]

        # img1 = img.transpose(1, 2, 0)
        # plt.figure(1)
        # plt.imshow(img1)
        # plt.show()
        img = torch.Tensor(img)

        label = np.zeros(10)
        label[self.dataset[item][1]] = 1
        label = torch.Tensor(label)
        label = torch.argmax(label, -1)
        img = img.to(self.device)
        label = label.to(self.device)
        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, args, dataset, target, portion, mode, idx, epoch):
        print("Generating " + mode + " Bad Imgs")
        # np.random.permutation（）总体来说他是一个随机排列函数，就是将输入的数据进行随机排列，官方文档指出，此函数只能针对一维数据随机排列
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * portion)]
        # len(perm),len(dataset):train:900 3000/test:3000 3000

        dataset_ = list()
        cnt = 0
        for i in range(len(dataset)):
            data = dataset[i]
            img = np.array(data[0])
            # print(np.max(img))
            img = np.resize(img, (1, 28, 28))
            channels = img.shape[0]
            width = img.shape[1]
            height = img.shape[2]
            pixel_value = 2.82
            distance = 3

            if i in perm:
                for c in range(channels):
                    if idx==args.attacker_list[0]:
                        img[c, 0, 0] = pixel_value
                        img[c, 0, 1] = pixel_value
                        img[c, 0, 2] = pixel_value
                        img[c, 0, 3] = pixel_value
                        print("0000000000000000")
                        plt.imshow(img.squeeze(0))
                        plt.show()

                    elif idx==args.attacker_list[1]:
                        img[c, 0, 6] = pixel_value
                        img[c, 0, 7] = pixel_value
                        img[c, 0, 8] = pixel_value
                        img[c, 0, 9] = pixel_value
                        print("11111111111111111")
                        plt.imshow(img.squeeze(0))
                        plt.show()

                    elif idx==args.attacker_list[2]:
                        img[c, 3, 0] = pixel_value
                        img[c, 3, 1] = pixel_value
                        img[c, 3, 2] = pixel_value
                        img[c, 3, 3] = pixel_value
                        print("222222222222222")
                        plt.imshow(img.squeeze(0))
                        plt.show()

                    elif idx==args.attacker_list[3]:
                        img[c, 3, 6] = pixel_value
                        img[c, 3, 7] = pixel_value
                        img[c, 3, 8] = pixel_value
                        img[c, 3, 9] = pixel_value
                        print("333333333333333")
                        plt.imshow(img.squeeze(0))
                        plt.show()

                    else:
                        img[c, 0, 0] = pixel_value
                        img[c, 0, 1] = pixel_value
                        img[c, 0, 2] = pixel_value
                        img[c, 0, 3] = pixel_value
                        img[c, 0, 6] = pixel_value
                        img[c, 0, 7] = pixel_value
                        img[c, 0, 8] = pixel_value
                        img[c, 0, 9] = pixel_value
                        img[c, 3, 0] = pixel_value
                        img[c, 3, 1] = pixel_value
                        img[c, 3, 2] = pixel_value
                        img[c, 3, 3] = pixel_value
                        img[c, 3, 6] = pixel_value
                        img[c, 3, 7] = pixel_value
                        img[c, 3, 8] = pixel_value
                        img[c, 3, 9] = pixel_value
                        print("44444444444444")
                        plt.imshow(img.squeeze(0), cmap='gray')
                        plt.show()

                dataset_.append((img, target))
                cnt += 1
            else:
                dataset_.append((img, data[1]))

        print("Injecting Over: " + str(cnt) + " Bad Imgs, " + str(len(dataset) - cnt) + " Clean Imgs")
        return dataset_

class MyDataset_cifar_DBA(Dataset):
    def __init__(self, args, dataset, target, portion=0.5, mode="train", idx=2, device='cpu'):
        self.dataset = self.addTrigger(args, dataset, target, portion, mode, idx)
        self.device = device

    def __getitem__(self, item):
        img = self.dataset[item][0]

        # img1 = img.transpose(1, 2, 0)
        # plt.figure(1)
        # plt.imshow(img1)
        # plt.show()

        # img = img[..., np.newaxis]
        #img = np.expand_dims(img, axis=0)
        # img = torch.Tensor(img).permute(2,0,1)
        img = torch.Tensor(img)

        label = np.zeros(10)
        label[self.dataset[item][1]] = 1
        label = torch.Tensor(label)
        label = torch.argmax(label, -1)
        img = img.to(self.device)
        label = label.to(self.device)
        return img, label

    def __len__(self):
        return len(self.dataset)


    def addTrigger(self, args, dataset, target, portion, mode, idx):
        print("Generating " + mode + " Bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * portion)]
        dataset_ = list()
        cnt = 0
        for i in range(len(dataset)):
            data = dataset[i]
            img = np.array(data[0])
            # print(np.max(img))
            img = np.resize(img, (3, 32, 32))
            channels = img.shape[0]
            width = img.shape[1]
            height = img.shape[2]
            pixel_value = 1
            distance = 3
            if i in perm:
                for c in range(channels):
                    if idx == args.attacker_list[0]:
                        img[c, 0, 0] = pixel_value
                        img[c, 0, 1] = pixel_value
                        img[c, 0, 2] = pixel_value
                        img[c, 0, 3] = pixel_value
                    elif idx == args.attacker_list[1]:
                        img[c, 0, 6] = pixel_value
                        img[c, 0, 7] = pixel_value
                        img[c, 0, 8] = pixel_value
                        img[c, 0, 9] = pixel_value
                    elif idx == args.attacker_list[2]:
                        img[c, 3, 0] = pixel_value
                        img[c, 3, 1] = pixel_value
                        img[c, 3, 2] = pixel_value
                        img[c, 3, 3] = pixel_value
                    elif idx == args.attacker_list[3]:
                        img[c, 3, 6] = pixel_value
                        img[c, 3, 7] = pixel_value
                        img[c, 3, 8] = pixel_value
                        img[c, 3, 9] = pixel_value
                    else:
                        img[c, 0, 0] = pixel_value
                        img[c, 0, 1] = pixel_value
                        img[c, 0, 2] = pixel_value
                        img[c, 0, 3] = pixel_value
                        img[c, 0, 6] = pixel_value
                        img[c, 0, 7] = pixel_value
                        img[c, 0, 8] = pixel_value
                        img[c, 0, 9] = pixel_value
                        img[c, 3, 0] = pixel_value
                        img[c, 3, 1] = pixel_value
                        img[c, 3, 2] = pixel_value
                        img[c, 3, 3] = pixel_value
                        img[c, 3, 6] = pixel_value
                        img[c, 3, 7] = pixel_value
                        img[c, 3, 8] = pixel_value
                        img[c, 3, 9] = pixel_value

                dataset_.append((img, target))
                cnt += 1
            else:
                dataset_.append((img, data[1]))

        print("Injecting Over: " + str(cnt) + " Bad Imgs, " + str(len(dataset) - cnt) + " Clean Imgs")
        return dataset_