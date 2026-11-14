import os
import time
import shutil
import logging

import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from src.models import AlexNet, \
    CNNCifar10, \
    CNNFashion_Mnist, \
    CNNMnist, \
    LeNet5, \
    CNNModel

from src.models import get_output_shape_of_last_layer
import numpy as np

########################################################################################################################
# Helper methods
########################################################################################################################

def get_torch_loss_function(name_of_loss_function):
    """
     Loads the required loss function object from the torch.nn library.
     If the library does not have the required loss function it throws an AttributeError.

     Parameters:
         name_of_loss_function: name of the required loss function.
     """
    if not hasattr(nn, name_of_loss_function):
        error_message = f"...Loss Function: \"{name_of_loss_function}\" is not supported or cannot be found in " \
                        f"Torch Optimizers! "
        logging.error(error_message)
        raise AttributeError(error_message)
    else:
        return nn.__dict__[name_of_loss_function]()


def create_one_hot_encoding(model):
    """
        Creates a one-hot encoding object based on the given model.

        Parameters:
            model: model to base the one hot encoding on.
     """
    output_size = get_output_shape_of_last_layer(model)
    return f.one_hot(input=torch.arange(0, output_size), num_classes=output_size)


def encode_labels(labels, encoding, device):
    """
     Encodes the given labels according to the given one-hot-encoding.

     Parameters:
         labels: Labels to be encoded.
         encoding: one-hot encoding used for transforming the labels.
         device: Device to simulate the models on.
     """
    labels = labels.type(torch.int64).cpu().numpy()
    return torch.stack(list(map(lambda x: encoding[x], labels))).to(device)


def get_duration(start_time):
    """
     Transforms a time interval format from seconds into hours, minutes and seconds.

     Parameters:
         start_time: time stamp used to define the time interval.
     """
    hours, rem = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    """
    TODO

    """
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def save_checkpoint_adversary(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    """ Ja hier moet dus documentatie """
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_adversary_best.pth.tar'))


def load_target_model(path_to_model_checkpoint, filename,model_name,num_class):
    """
     Loads the target model from a previous checkpoint to resume training or
     to be used to generate input for the attack model.

     Note this has hardcoded in to load the AlexNet model from src.models.

     Parameters:
         path_to_model_checkpoint: path to the location of the model checkpoint.
         filename: name of the model checkpoint file.
     """
    #################
    #need to do
    #################
    if model_name=="AlexNet":
        model = AlexNet(num_class)
    elif model_name=="CNNCifar10":
        model=CNNCifar10(num_class)
    elif model_name=="CNNFashion_Mnist":
        model=CNNFashion_Mnist(num_class)
    elif model_name=="CNNMnist":
        model=CNNMnist(num_class)
    elif model_name=="LeNet5":
        model=LeNet5(num_class)
    elif model_name == "CNNModel":
        model = CNNModel()
    filepath = os.path.join(path_to_model_checkpoint, filename)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def load_dataset(path_for_data_set, data_set_name):
    """
     Main method for loading the data set. If the name equals a pre-defined string use the customs settings.
     Otherwise, try to find the data set in the torchvision.datasets library. If not present, throw AttributeError.

     Parameters:
         path_for_data_set: String containing the path to the location of the data set.
         data_set_name: Name of the required data set.
     """
    logging.debug('Loading datasets..')
    # If True, use custom settings for this specific data set.
    if data_set_name == 'tiny-imagenet-100':
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2296, 0.2263, 0.2255)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2296, 0.2263, 0.2255)),
        ])

        train_set = datasets.ImageFolder(root=path_for_data_set + data_set_name + '/train',
                                         transform=transform_train
                                         )

        test_set = datasets.ImageFolder(root=path_for_data_set + data_set_name + '/test',
                                        transform=transform_test
                                        )

        logging.debug('Loading completed')

        return train_set, test_set
    elif data_set_name == 'cifar-100-python':
        # prepare raw training & test datasets
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        training_data = datasets.CIFAR100(
            root=path_for_data_set,
            train=True,
            download=True,
            transform=transform_train,

        )

        test_data = datasets.CIFAR100(
            root=path_for_data_set,
            train=False,
            download=True,
            transform=transform_test
        )
        logging.debug('Loading completed')

        return training_data, test_data
    elif data_set_name=="cifar10":
        # prepare raw training & test datasets
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        training_data = datasets.CIFAR10(
            root=path_for_data_set,
            train=True,
            download=True,
            transform=transform_train
        )

        test_data = datasets.CIFAR10(
            root=path_for_data_set,
            train=False,
            download=True,
            transform=transform_test
        )
        logging.debug('Loading completed')

        return training_data, test_data
    elif data_set_name=="mnist":
        # prepare raw training & test datasets
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        training_data = datasets.MNIST(root=path_for_data_set,
                                       train=True,
                                       download=True,
                                       transform=transform_train)

        test_data = datasets.MNIST(
            root=path_for_data_set,
            train=False,
            download=True,
            transform=transform_test
        )
        logging.debug('Loading completed')

        return training_data, test_data
    elif data_set_name=="fashion_mnist":
        # prepare raw training & test datasets
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        training_data = datasets.FashionMNIST(
            root=path_for_data_set,
            train=True,
            download=True,
            transform=transform_train
        )

        test_data = datasets.FashionMNIST(
            root=path_for_data_set,
            train=False,
            download=True,
            transform=transform_test
        )
        logging.debug('Loading completed')
        return training_data, test_data
    else:
        #在torchvision.datasets中尋找數據集
        if not hasattr(torchvision.datasets, data_set_name):
            # dataset not found exception
            error_message = f"...dataset \"{data_set_name}\" is not supported or cannot be found in TorchVision " \
                            f"Datasets! "
            logging.error(error_message)
            raise AttributeError(error_message)
        else:
            # prepare raw training & test datasets
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])

            training_data = torchvision.datasets.__dict__[data_set_name](
                root=path_for_data_set,
                train=True,
                download=True,
                transform=transform_train
            )

            test_data = torchvision.datasets.__dict__[data_set_name](
                root=path_for_data_set,
                train=False,
                download=True,
                transform=transform_test
            )
            logging.debug('Loading completed')

        return training_data, test_data


'''
def cifar_100_noniid(dataset, num_users):
    """
    num:50000
    相当与把数据集按类别分类，0--10，然后每个用户选取n个切片，每个切片是连续的num_imgs张类的图片
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # # 60,000 training imgs -->  200 imgs/shard X 300 shards
    # num_shards, num_imgs = 200, 300
    num_shards, num_imgs = 1000, 50
    #if args.dataset == 'mnist':
    #    num_shards, num_imgs = 1000, 50

    #if args.dataset == 'fmnist':
    #    num_shards, num_imgs = 1000, 59
    #1000 个数据切片 一个切片里有50条数据,
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets

    # sort labels
    idxs_labels = np.vstack((idxs, labels))#数据下标和标签对应拼接在一起

    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]#按照标签（类别顺序）进行排列从第0类到第100类所对应的下标
    idxs = idxs_labels[0, :]#取下标列

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_shards//num_users, replace=False))#从1200个切片中选择num_shards//num_users个切片不重复
        idx_shard = list(set(idx_shard) - rand_set)#确保idx_shard里没有已经分配过的切片
        for rand in rand_set:
            # 第i个客户
            dict_users[i] = np.concatenate(#拼接起来
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    for i in range(num_users):
        dict_users[i]=dict_users[i].astype(int)
    return dict_users


def cifar_10_noniid(dataset, num_users,cl):
    """
    num:50000
    相当与把数据集按类别分类，0--10，然后每个用户选取n个切片，每个切片是连续的num_imgs张类的图片
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 1000, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = dataset.targets

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_shards // num_users, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    for i in range(num_users):
        dict_users[i]=dict_users[i].astype(int)
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    num:60000
    相当与把数据集按类别分类，0--10，然后每个用户选取n个切片，每个切片是连续的num_imgs张类的图片
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # # 60,000 training imgs -->  200 imgs/shard X 300 shards
    # num_shards, num_imgs = 200, 300
    num_shards, num_imgs = 1200, 50

    #1200 个数据切片 一个切片里有50条数据
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets

    # sort labels
    idxs_labels = np.vstack((idxs, labels))#数据下标和标签对应拼接在一起

    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]#按照标签（类别顺序）进行排列从第0类到第100类所对应的下标
    idxs = idxs_labels[0, :]#取下标列

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_shards//num_users, replace=False))#从1200个切片中选择num_shards//num_users个切片不重复
        idx_shard = list(set(idx_shard) - rand_set)#确保idx_shard里没有已经分配过的切片
        for rand in rand_set:
            # 第i个客户
            dict_users[i] = np.concatenate(#拼接起来
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    for i in range(num_users):
        dict_users[i]=dict_users[i].astype(int)

    return dict_users


def fashion_mnist_noniid(dataset, num_users):
    """
    num:60000
    相当与把数据集按类别分类，0--10，然后每个用户选取n个切片，每个切片是连续的num_imgs张类的图片
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # # 60,000 training imgs -->  200 imgs/shard X 300 shards
    # num_shards, num_imgs = 200, 300
    user_class_num=10/num_users
    num_shards=user_class_num
    num_imgs = len(dataset)//num_shards

    #1200 个数据切片 一个切片里有50条数据
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets

    # sort labels
    idxs_labels = np.vstack((idxs, labels))#数据下标和标签对应拼接在一起

    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]#按照标签（类别顺序）进行排列从第0类到第100类所对应的下标
    idxs = idxs_labels[0, :]#取下标列

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_shards//num_users, replace=False))#从1200个切片中选择num_shards//num_users个切片不重复
        idx_shard = list(set(idx_shard) - rand_set)#确保idx_shard里没有已经分配过的切片
        for rand in rand_set:
            # 第i个客户
            dict_users[i] = np.concatenate(#拼接起来
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    for i in range(num_users):
        dict_users[i]=dict_users[i].astype(int)
    return dict_users

'''







################################################################
def get_noniid_datadict(dataset_name,dataset, num_users, num_class,isoverlap,num_user_class):
    client_img_nums = len(dataset) // num_users

    client_each_class_num = client_img_nums // num_user_class
    class_num = int(len(dataset) / num_class)
    dict_users_class = {i: [] for i in range(num_users)}  # each client classs
    dict_users = {i: [] for i in range(num_users)}  # each client index
    idx_class = [i for i in range(num_class)]
    weights = [class_num // client_each_class_num for i in range(num_class)]
    weights = [weights[i] / sum(weights) for i in range(num_class)]
    #print(weights)

    idx_class_times = []
    for i in range(num_class):
        for j in range(class_num // client_each_class_num):
            idx_class_times.append(i)

    #print(idx_class)
    #print(idx_class_times)
    # sort labels
    idxs = np.arange(len(dataset))
    labels = dataset.targets
    idxs_labels = np.vstack((idxs, labels))  # 数据下标和标签对应拼接在一起
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 按照标签（类别顺序）进行排列从第0类到第10类所对应的下标
    idxs = idxs_labels[0, :]  # 取下标列
    # idx_left = idx_shard# left indexs
    if dataset_name=='mnist':
        class_index_list=[]
        class_index_list.append([i for i in range(0, 5923)])
        class_index_list.append([i for i in range(5923, 12665)])
        class_index_list.append([i for i in range(12665, 18623)])
        class_index_list.append([i for i in range(18623, 24753)])
        class_index_list.append([i for i in range(24753, 30596)])
        class_index_list.append([i for i in range(30596, 36017)])
        class_index_list.append([i for i in range(36017, 41936)])
        class_index_list.append([i for i in range(41936, 48120)])
        class_index_list.append([i for i in range(48120, 54051)])
        class_index_list.append([i for i in range(54051, 60000)])
    else:
        class_index_list = [np.arange(i * class_num, (i + 1) * class_num) for i in range(num_class)]

    #print(class_index_list)
    #print(idxs)

    for i in range(num_users):
        if i ==num_users-1:
            rand_set=idx_class_times
            idx_class_times=[]
        else :
            rand_set = set(np.random.choice(idx_class, size=num_user_class, replace=False,
                                        p=weights))  # chose class_index from classnums and each client no repeat suach as no {1,1,2}
            for class_index in (rand_set):
                idx_class_times.remove(class_index)
        #rand_set = set(np.random.choice(idx_class_times, size=num_user_class, replace=False))  # chose class_index from classnums and each client no repeat suach as no {1,1,2}

        #print(rand_set)

        sum_idx_class_times = len(idx_class_times)

        for class_index in range(num_class):
            count = 0;
            for j in idx_class_times:
                if j == class_index: count += 1
            if sum_idx_class_times == 0:
                weights[class_index] = 0
            else:
                weights[class_index] = count / sum_idx_class_times
        #print(weights)
        for class_index in rand_set:
            dict_users_class[i].append(class_index)
    #print(dict_users_class)

    for i in range(num_users):
        for class_index in dict_users_class[i]:
            class_list = class_index_list[class_index]
            # print(class_list)
            if client_each_class_num>len(class_list):
                print("用户"+str(i)+"使用类别"+str(class_index)+"时，该类数据不够均分从"+str(client_each_class_num)+"到"+str(len(class_list))+"\n")
                client_each_class_num=len(class_list)
            rand_set = set(np.random.choice(class_list, client_each_class_num, replace=False))
            # print(i,' ',class_index,' ',len(rand_set))
            #temp = [len(class_index_list[i]) for i in range(num_class)]
            #print(temp)
            if isoverlap == 0:
                class_index_list[class_index] = list(set(class_index_list[class_index]) - rand_set)
            for index in rand_set:
                dict_users[i].append(idxs[index])
    #print(dict_users_class)
    #print('sdsdsd')
    return dict_users

################################################################




'''
def get_noniid_datadict(dataset, num_users, num_class,isoverlap,num_user_class):
    """
    num:60000
    相当与把数据集按类别分类，0--num_class，然后每个用户选取num_class/num_users个切片，每个切片是连续的len(dataset)/num_imgs张类的图片
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # # 60,000 training imgs -->  200 imgs/shard X 300 shards
    # num_shards, num_imgs = 200, 300

    # print("num_shards//num_users",num_class//num_users) #2
    #each client img nums
    client_img_nums=len(dataset)//num_users
    client_each_class_num=client_img_nums//num_user_class






    num_shards=num_class
    num_imgs = len(dataset)//num_shards
    # print("num_imgs",num_imgs) #6000 #每个类有多少样本

    # num_shards 个数据切片 一个切片里有len(dataset)//num_shards条数据: 10个数据切片，一个切片里有6000条数据
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets

    # sort labels
    idxs_labels = np.vstack((idxs, labels))#数据下标和标签对应拼接在一起

    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]#按照标签（类别顺序）进行排列从第0类到第10类所对应的下标
    idxs = idxs_labels[0, :]#取下标列
    idx_shard_left=idx_shard
    # divide and assign 2 shards/client

    # print("num_shards//num_users", num_shards// num_users)
    for i in range(num_users):
        # print("i",i)
        rand_set = set(np.random.choice(idx_shard, num_shards//num_users, replace=False))#从10个切片中选择2个切片不重复
        # print("rand_set ",rand_set) #随机选取两个切片
        if isoverlap==0:
            idx_shard = list(set(idx_shard) - rand_set)#确保idx_shard里没有已经分配过的切片
            # print("idx_shard",idx_shard) #剩下的切片

        idx_shard_left=list(set(idx_shard_left) - rand_set)
        # print("idx_shard_left",idx_shard_left)

        for rand in rand_set:
            # 第i个客户
            # print("rand",rand) #每个切片
            dict_users[i] = np.concatenate(#拼接起来
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
            # print("dict_users[i] ",len(dict_users[i]))

    #如果有剩餘的把剩餘的按順序分給每個用戶
    if len(idx_shard_left)!=0:
        for idx in range(num_shards%num_users):
            index=idx_shard_left[idx]
            low=index*num_imgs
            length=num_imgs//num_users
            for i in range(num_users):
                dict_users[i] = np.concatenate(  # 拼接起来
                    (dict_users[i], idxs[low :low+length]), axis=0)
                low+=length

    for i in range(num_users):
        dict_users[i]=dict_users[i].astype(int)
    return dict_users
'''

########################################################################################################################
# Classes for ConfusionMatrix and AverageMeter, used to track metrics during runtime.
########################################################################################################################

class ConfusionMatrix(object):
    """
    ConfusionMatrix Class
    Used to keep track of True/False positives and True/False negatives.

    Methods:
        reset: method to reset all attributes to zero.
        update_from_matrix: update all attributes with the scores from another ConfusionMatrix object.
        update: update all attributes by comparing the predictions and the labels.
        get_accuracy: calculate the accuracy based on the attributes.
        get_confusion_matrix: Getter method for the attributes of the Confusion Matrix.


     """

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update_from_matrix(self, confusion_matrix):
        (tp, fp, tn, fn) = confusion_matrix

        self.tp += tp
        self.fn += fn
        self.fp += fp
        self.tn += tn

    def update(self, predictions, labels):
        confusion_vector = (torch.round(predictions) / labels).float()

        self.tp += torch.sum(confusion_vector == 1.).item()
        self.fn += torch.sum(confusion_vector == 0.).item()
        self.fp += torch.sum(confusion_vector == float('inf')).item()
        self.tn += torch.sum(torch.isnan(confusion_vector)).item()

    def get_accuracy(self):
        if (self.tp + self.tn + self.fp + self.fn)==0:
            return 0
        return ((self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)) * 100

    def get_confusion_matrix(self):
        return self.tp, self.fp, self.tn, self.fn


class AverageMeter(object):
    """
        Computes and stores the average and current value
        Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.total = None
        self.sum = None
        self.avg = None
        self.correct = None
        self.reset()

    def reset(self):
        self.correct = 0
        self.avg = 0
        self.sum = 0
        self.total = 0

    def update(self, correct, n=1):
        self.correct = correct
        self.sum += correct * n
        self.total += n
        self.avg = self.sum / self.total
