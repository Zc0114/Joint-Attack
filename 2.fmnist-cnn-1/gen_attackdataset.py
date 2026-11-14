import copy
import logging
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import yaml

from src.attacker import Attacker
from src.client import Client
from src.models import AlexNet, \
    CNNCifar10, \
    CNNFashion_Mnist, \
    CNNMnist, \
    LeNet5
from src.utils import load_dataset, \
    AverageMeter, \
    get_torch_loss_function, \
    load_target_model, \
    save_checkpoint, \
    save_checkpoint_adversary, \
    get_noniid_datadict

from src.patch_utils import *
from src.util import *
from torch.autograd import Variable
from torch import nn, autograd
from src.backdoor_data import MyDataset_cifar, MyDataset_mnist, MyDataset_cifar_DBA, MyDataset_mnist_DBA, MyDataset_cifar_new



def main():
    with open('./config.yaml') as c:
        configs = list(yaml.load_all(c, Loader=yaml.FullLoader))

    experiment_config = configs[0]["experiment_config"]
    attack_config = configs[1]['attack_settings']
    data_config = configs[3]["data_settings"]

    distribution = attack_config['attack_data_distribution']
    number_of_classes=data_config['number_of_classes']
    # print("len(client_indices)",len(client_indices)) #50000
    # print("client_dataset_size",client_dataset_size) #12500
    dataset_path = data_config['data_path']
    dataset_name = data_config['dataset_name']

    train_data, test_data = load_dataset(dataset_path, dataset_name)


    # check is data sizes are correct:

    #assert len(client_indices) >= client_dataset_size
    #assert len(client_indices) >= distribution[0] + distribution[1]

    # num of member and non member for train attack need increase
    # train_member_length = distribution[0] // 100
    # train_non_member_length = distribution[1] // 100
    # test_member_length = distribution[2] // 100
    # test_non_member_length = distribution[3] // 100
    train_member_length = distribution[0] // number_of_classes
    train_non_member_length = distribution[1] // number_of_classes
    test_member_length = distribution[2] // number_of_classes
    test_non_member_length = distribution[3] // number_of_classes
    # print("train_member_length",train_member_length) #20

    # Set index lists to empty list.
    training_member_index, training_non_member_index, test_member_index, test_non_member_index = [], [], [], []

    ##################################sort dataset by label
    train_data_sort_indices = sorted(range(len(train_data)), key=lambda i: train_data[i][1])  # 按标签排序的索引
    train_data_sort = Subset(train_data, train_data_sort_indices)

    test_data_sort_indices = sorted(range(len(test_data)), key=lambda i: test_data[i][1])  # 按标签排序的索引
    test_data_sort = Subset(test_data, test_data_sort_indices)

    train_class_num = len(train_data) // number_of_classes
    test_class_num = len(test_data) // number_of_classes
    ##################################

    # Get the indicis for the right amount of training and test samples.
    # may want to take same number from each class
    for target in range(number_of_classes):
        # Set the boundaries for picking samples from the right classes
        low_member_train, high_member_train = target * train_class_num, (target + 1) * train_class_num
        low_non_member_train, high_non_member_train = target * train_class_num, (target + 1) * train_class_num

        low_member_test, high_member_test = target * test_class_num, (target + 1) * test_class_num
        low_non_member_test, high_non_member_test = target * test_class_num, (target + 1) * test_class_num
        low_train_backdoor_gradascent, high_train_backdoor_gradascent = target * train_class_num, (
                    target + 1) * train_class_num
        random_subset = list(range(low_member_train,high_member_train))
        random.shuffle(random_subset)

        for _ in range(train_member_length):
            training_member_index.append(random_subset.pop())

        for _ in range(test_member_length):
            test_member_index.append(random_subset.pop())

        # random_subset = list(range(low_non_member, high_non_member))
        ##########################
        # random_subset = list(
        #    [x for x in range(low_non_member_train, high_non_member_train) if x not in client_indices])

        random_subset = list(range(low_non_member_test, high_non_member_test))

        ##########################
        random.shuffle(random_subset)

        for _ in range(train_non_member_length):
            training_non_member_index.append(random_subset.pop())

        for _ in range(test_non_member_length):
            test_non_member_index.append(random_subset.pop())

    train_member = torch.utils.data.Subset(train_data_sort, training_member_index)
    train_non_member = torch.utils.data.Subset(test_data_sort, training_non_member_index)
    test_member = torch.utils.data.Subset(train_data_sort, test_member_index)
    test_non_member = torch.utils.data.Subset(test_data_sort, test_non_member_index)
    # attack_train_data_member and attack_train_non_member get value

    attack_train_member = train_member
    attack_train_nonmember = train_non_member
    attack_test_member=test_member
    attack_test_nonmember = test_non_member

    train_member_length = distribution[0]
    train_non_member_length = distribution[1]
    test_member_length = distribution[2]
    test_non_member_length = distribution[3]
    if dataset_name=='cifar10':
        attack_train_member_path='./cifar10data/attack_train_member_'+str(train_member_length)+'.pth'
        attack_train_nonmember_path='./cifar10data/attack_train_nonmember_'+str(train_non_member_length)+'.pth'
        attack_test_member_path='./cifar10data/attack_test_member_'+str(test_member_length)+'.pth'
        attack_test_nonmember_path='./cifar10data/attack_test_nonmember_'+str(test_non_member_length)+'.pth'
    elif dataset_name=='fashion_mnist':
        attack_train_member_path='./fashion_mnistdata/attack_train_member_'+str(train_member_length)+'.pth'
        attack_train_nonmember_path='./fashion_mnistdata/attack_train_nonmember_'+str(train_non_member_length)+'.pth'
        attack_test_member_path='./fashion_mnistdata/attack_test_member_'+str(test_member_length)+'.pth'
        attack_test_nonmember_path='./fashion_mnistdata/attack_test_nonmember_'+str(test_non_member_length)+'.pth'
    elif dataset_name=='mnist':
        attack_train_member_path='./mnistdata/attack_train_member_'+str(train_member_length)+'.pth'
        attack_train_nonmember_path='./mnistdata/attack_train_nonmember_'+str(train_non_member_length)+'.pth'
        attack_test_member_path='./mnistdata/attack_test_member_'+str(test_member_length)+'.pth'
        attack_test_nonmember_path='./mnistdata/attack_test_nonmember_'+str(test_non_member_length)+'.pth'
    else:
        attack_train_member_path='./cifar100data/attack_train_member_'+str(train_member_length)+'.pth'
        attack_train_nonmember_path='./cifar100data/attack_train_nonmember_'+str(train_non_member_length)+'.pth'
        attack_test_member_path='./cifar100data/attack_test_member_'+str(test_member_length)+'.pth'
        attack_test_nonmember_path='./cifar100data/attack_test_nonmember_'+str(test_non_member_length)+'.pth'


    torch.save(attack_train_member,attack_train_member_path)
    torch.save(attack_train_nonmember, attack_train_nonmember_path)
    torch.save(attack_test_member, attack_test_member_path)
    torch.save(attack_test_nonmember, attack_test_nonmember_path)

    print("train_member", len(train_member))  # 2000
    if dataset_name=='cifar-100-python':
        temp = []
        for i in range(100):
            temp.append(0)
    else:
        temp = []
        for i in range(10):
            temp.append(0)

    for item in attack_train_member:
        temp[item[1]] += 1
    print(temp)

    print("test_member", len(test_member))  # 2000
    if dataset_name=='cifar-100-python':
        temp = []
        for i in range(100):
            temp.append(0)
    else:
        temp = []
        for i in range(10):
            temp.append(0)

    for item in attack_test_member:
        temp[item[1]] += 1
    print(temp)


    print("train_non_member", len(train_non_member))  # 2000
    if dataset_name=='cifar-100-python':
        temp = []
        for i in range(100):
            temp.append(0)
    else:
        temp = []
        for i in range(10):
            temp.append(0)

    for item in attack_train_nonmember:
        temp[item[1]] += 1
    print(temp)

    print("test_non_member", len(test_non_member))  # 2000
    if dataset_name=='cifar-100-python':
        temp = []
        for i in range(100):
            temp.append(0)
    else:
        temp = []
        for i in range(10):
            temp.append(0)

    for item in attack_test_nonmember:
        temp[item[1]] += 1
    print(temp)





if __name__ == '__main__':
    main()