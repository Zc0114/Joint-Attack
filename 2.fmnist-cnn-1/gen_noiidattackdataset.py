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
    training_config = configs[2]["training_settings"]

    distribution = attack_config['attack_data_distribution']
    number_of_classes=data_config['number_of_classes']

    no_iid_class_num=training_config['no_iid_class_num']

    dataset_path = data_config['data_path']
    dataset_name = data_config['dataset_name']
    num_class=data_config['number_of_classes']

    train_data, test_data = load_dataset(dataset_path, dataset_name)
    idx_class=idx_class = [i for i in range(num_class)]

    rand_set_member = set(np.random.choice(idx_class, size=no_iid_class_num, replace=False))
    rand_set_nonmember=set(np.random.choice(idx_class, size=no_iid_class_num, replace=False))


    train_member_length = distribution[0] // no_iid_class_num
    train_non_member_length = distribution[1] // no_iid_class_num
    test_member_length = distribution[2] // no_iid_class_num
    test_non_member_length = distribution[3] // no_iid_class_num


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

    if dataset_name=='mnist':
        class_index_list_member=[]
        class_index_list_member.append([i for i in range(0, 5923)])
        class_index_list_member.append([i for i in range(5923, 12665)])
        class_index_list_member.append([i for i in range(12665, 18623)])
        class_index_list_member.append([i for i in range(18623, 24753)])
        class_index_list_member.append([i for i in range(24753, 30596)])
        class_index_list_member.append([i for i in range(30596, 36017)])
        class_index_list_member.append([i for i in range(36017, 41936)])
        class_index_list_member.append([i for i in range(41936, 48120)])
        class_index_list_member.append([i for i in range(48120, 54051)])
        class_index_list_member.append([i for i in range(54051, 60000)])
    else:
        class_index_list_member = [np.arange(i * train_class_num, (i + 1) * train_class_num) for i in range(num_class)]

    if dataset_name=='mnist':
        class_index_list_nonmember=[]
        class_index_list_nonmember.append([i for i in range(0, 981)])
        class_index_list_nonmember.append([i for i in range(981, 2117)])
        class_index_list_nonmember.append([i for i in range(2117, 3149)])
        class_index_list_nonmember.append([i for i in range(3149, 4159)])
        class_index_list_nonmember.append([i for i in range(4159, 5141)])
        class_index_list_nonmember.append([i for i in range(5141, 6033)])
        class_index_list_nonmember.append([i for i in range(6033, 6991)])
        class_index_list_nonmember.append([i for i in range(6991, 8019)])
        class_index_list_nonmember.append([i for i in range(8019, 8993)])
        class_index_list_nonmember.append([i for i in range(8993, 10000)])
    else:
        class_index_list_nonmember = [np.arange(i * test_class_num, (i + 1) * test_class_num) for i in range(num_class)]

    idxs = np.arange(len(train_data))
    labels = train_data.targets
    idxs_labels = np.vstack((idxs, labels))  # 数据下标和标签对应拼接在一起
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 按照标签（类别顺序）进行排列从第0类到第10类所对应的下标
    idxs = idxs_labels[0, :]  # 取下标列


    for class_index in rand_set_member:
        class_list = class_index_list_member[class_index]
        # print(class_list)
        rand_set = set(np.random.choice(class_list, train_member_length, replace=False))
        # print(i,' ',class_index,' ',len(rand_set))
        # temp = [len(class_index_list[i]) for i in range(num_class)]
        # print(temp)

        for index in rand_set:
            training_member_index.append(idxs[index])

        class_index_list_member[class_index] = list(set(class_index_list_member[class_index]) - rand_set)

        class_list = class_index_list_member[class_index]
        # print(class_list)
        rand_set = set(np.random.choice(class_list, test_member_length, replace=False))
        # print(i,' ',class_index,' ',len(rand_set))
        # temp = [len(class_index_list[i]) for i in range(num_class)]
        # print(temp)

        for index in rand_set:
            test_member_index.append(idxs[index])

        class_index_list_member[class_index] = list(set(class_index_list_member[class_index]) - rand_set)


    idxs_nonmember = np.arange(len(test_data))
    labels_nonmember = test_data.targets
    idxs_labels_nonmember = np.vstack((idxs_nonmember, labels_nonmember))  # 数据下标和标签对应拼接在一起
    idxs_labels_nonmember = idxs_labels_nonmember[:, idxs_labels_nonmember[1, :].argsort()]  # 按照标签（类别顺序）进行排列从第0类到第10类所对应的下标
    idxs_nonmember = idxs_labels_nonmember[0, :]  # 取下标列

    for class_index in rand_set_nonmember:
        class_list = class_index_list_nonmember[class_index]
        # print(class_list)
        rand_set = set(np.random.choice(class_list, train_non_member_length, replace=False))
        # print(i,' ',class_index,' ',len(rand_set))
        # temp = [len(class_index_list[i]) for i in range(num_class)]
        # print(temp)

        for index in rand_set:
            training_non_member_index.append(idxs_nonmember[index])

        class_index_list_nonmember[class_index] = list(set(class_index_list_nonmember[class_index]) - rand_set)

        class_list = class_index_list_nonmember[class_index]
        # print(class_list)
        rand_set = set(np.random.choice(class_list, test_non_member_length, replace=False))
        # print(i,' ',class_index,' ',len(rand_set))
        # temp = [len(class_index_list[i]) for i in range(num_class)]
        # print(temp)

        for index in rand_set:
            test_non_member_index.append(idxs_nonmember[index])

        class_index_list_nonmember[class_index] = list(set(class_index_list_nonmember[class_index]) - rand_set)


    train_member = torch.utils.data.Subset(train_data, training_member_index)
    train_non_member = torch.utils.data.Subset(test_data, training_non_member_index)
    test_member = torch.utils.data.Subset(train_data, test_member_index)
    test_non_member = torch.utils.data.Subset(test_data, test_non_member_index)
    # attack_train_data_member and attack_train_non_member get value



    attack_train_member = train_member
    attack_train_nonmember = train_non_member
    attack_test_member=test_member
    attack_test_nonmember = test_non_member

    train_member_length = distribution[0]
    train_non_member_length = distribution[1]
    test_member_length = distribution[2]
    test_non_member_length = distribution[3]

    attack_train_member_path='./noiid/attack_train_member_'+str(train_member_length)+'.pth'
    attack_train_nonmember_path='./noiid/attack_train_nonmember_'+str(train_non_member_length)+'.pth'
    attack_test_member_path='./noiid/attack_test_member_'+str(test_member_length)+'.pth'
    attack_test_nonmember_path='./noiid/attack_test_nonmember_'+str(test_non_member_length)+'.pth'


    torch.save(attack_train_member,attack_train_member_path)
    torch.save(attack_train_nonmember, attack_train_nonmember_path)
    torch.save(attack_test_member, attack_test_member_path)
    torch.save(attack_test_nonmember, attack_test_nonmember_path)

    print("train_member", len(train_member))  # 2000
    temp = []
    for i in range(10):
        temp.append(0)

    for item in attack_train_member:
        temp[item[1]] += 1
    print(temp)

    print("test_member", len(test_member))  # 2000
    temp = []
    for i in range(10):
        temp.append(0)

    for item in attack_test_member:
        temp[item[1]] += 1
    print(temp)


    print("train_non_member", len(train_non_member))  # 2000
    temp = []
    for i in range(10):
        temp.append(0)

    for item in attack_train_nonmember:
        temp[item[1]] += 1
    print(temp)

    print("test_non_member", len(test_non_member))  # 2000
    temp = []
    for i in range(10):
        temp.append(0)
    for item in attack_test_nonmember:
        temp[item[1]] += 1
    print(temp)






if __name__ == '__main__':
    main()