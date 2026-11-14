import copy
import logging
import random
import time
from datetime import datetime
import yaml
from sklearn.manifold import TSNE
from scipy.spatial import distance
import torch
import torch.optim as optimizers
import torch.utils
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm
from src.models import AttackModel,AttackModel2
from matplotlib.pyplot import MultipleLocator
from src.models import get_last_layer_name
from src.utils import AverageMeter, get_torch_loss_function, ConfusionMatrix, encode_labels, create_one_hot_encoding
from .client import Client
from pylab import xticks,yticks
from src.patch_utils import *
from src.util import *
from torch.autograd import Variable
from torch import nn, autograd
from src.backdoor_data import MyDataset_cifar, MyDataset_mnist, MyDataset_cifar_DBA, MyDataset_mnist_DBA, MyDataset_cifar_new
import numpy as np
import os
from PIL import Image

class Attacker(Client):

    ####################################################################################################################
    def repixeo(self,img_array, min_value, max_value):
        scaled_img_array = (img_array - torch.min(img_array)) * (max_value - min_value) / (
                                        torch.max(img_array) - torch.min(img_array)) + min_value

        return scaled_img_array


    def patch_attack(self, image, applied_patch, mask, target, probability_threshold, model,use_mialoss_ratio,attmodelres, lr=1, max_iteration=100):
        model.eval()
        applied_patch = torch.from_numpy(applied_patch)
        mask = torch.from_numpy(mask)
        target_probability, count = 0, 0
        perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul(
            (1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
        
        while target_probability < probability_threshold and count < max_iteration:
            count += 1
            # Optimize the patch
            perturbated_image = Variable(perturbated_image.data, requires_grad=True)
            per_image = perturbated_image
            #if self.device == 'cuda':
            #    per_image = per_image.cuda()
            per_image=per_image.to(self.device)
            model=model.to(self.device)
            output = model(per_image)
            for param in model.parameters():
                param.grad = None
            #have target prob

            target_log_softmax = torch.nn.functional.log_softmax(output, dim=1)[0][target]+use_mialoss_ratio*attmodelres


            model=model.to(self.device)
            target_log_softmax.backward(retain_graph=True)
            patch_grad = perturbated_image.grad.clone().cpu()
            perturbated_image.grad.data.zero_()


            #grad img plus patch
            applied_patch = lr * patch_grad + applied_patch.type(torch.FloatTensor)

            #applied_patch = torch.clamp(applied_patch, min=-3, max=3)
            # Test the patch
            #patch added grad to img,grad from target class
            perturbated_image = torch.mul(mask.type(torch.FloatTensor),
                                          applied_patch.type(torch.FloatTensor)) + torch.mul(
                (1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
            perturbated_image = perturbated_image.to(self.device)
            applied_patch=applied_patch.to(self.device)
            image=image.to(self.device)
            perturbated_image = torch.clamp(perturbated_image, min=torch.min(image), max=torch.max(image))
            #applied_patch = torch.clamp(applied_patch, min=torch.min(image), max=torch.max(image))

            #if self.data_name=='mnist' or self.data_name=='fashion_mnist':

            #if self.device=='cuda':
            #    perturbated_image = perturbated_image.cuda()

            #perturbated_image=perturbated_image.to(self.device)
            output = model(perturbated_image)
            target_probability = torch.nn.functional.softmax(output, dim=1).data[0][target]
        perturbated_image = perturbated_image.cpu().numpy()
        #applied_patch=applied_patch.to(self.device)
        #applied_patch=self.repixeo(applied_patch,torch.min(image), torch.max(image))
        applied_patch = applied_patch.cpu().numpy()
        # 假设 perturbated_image 是一个 numpy 数组
        if self.data_name=='cifar10' or self.data_name=='cifar-100-python':
            perturbated_image_numpy = perturbated_image[0].transpose((1, 2, 0))  # 将通道维度移到最后
            perturbated_image_numpy = np.clip(perturbated_image_numpy, 0, 1)  # 将像素值限制在 0 到 1 之间
            perturbated_image_numpy = (perturbated_image_numpy * 255).astype(np.uint8)  # 将图像像素值转换为 0-255 范围内的整数，并转换数据类型为 uint8
            perturbated_image_pil = Image.fromarray(perturbated_image_numpy)  # 创建 PIL 图像对象
            perturbated_image_pil.save('perturbated_image.png')
        # else:
        #     perturbated_image_numpy =  perturbated_image[0].reshape((28, 28))
        #     perturbated_image_numpy = np.clip(perturbated_image_numpy, 0, 1)  # 将像素值限制在 0 到 1 之间
        #     perturbated_image_numpy = (perturbated_image_numpy * 255).astype(np.uint8)  # 将图像像素值转换为 0-255 范围内的整数，并转换数据类型为 uint8
        #     perturbated_image_pil = Image.fromarray(perturbated_image_numpy)  # 创建 PIL 图像对象
        # 保存图像

        return perturbated_image, applied_patch
    ####################################################################################################################

    def __init__(self, client_id,experiment_param,local_data, attack_data, backdoor_data, device, target_train_model, number_of_observed_models, number_of_classes, model_name, data_name,adv_patch_cpu):
        """
         Attacker class for membership inference attack.
         Based on the KERAS implementation of the paper by Naser et al.

         Paper link: https://arxiv.org/abs/1812.00910

         Implementation link: https://github.com/SPIN-UMass/MembershipWhiteboxAttacks/tree
         /fed4a30107393b42f955ca399c7d0df162e73eb3

         """
        super().__init__(client_id, local_data, device, target_train_model)
        self.eval_attack = attack_data['eval_attack']
        self.attack_train_batch_size = attack_data['train_batch_size']
        self.attack_test_batch_size = attack_data['test_batch_size']
        self.one_hot_encoding = create_one_hot_encoding(self.model)

        self.attack_data_distribution = attack_data['attack_data_distribution']
        self.attack_loss_function = get_torch_loss_function(attack_data['attack_loss_function'])
        self.attack_optimizer_name = attack_data['attack_optimizer']
        self.attack_model_learning_rate = attack_data['attack_model_learning_rate']
        # self.attack_model_learning_rate = attack_data['active_learning_rate']

        # Pre define general variables.
        self.class_test_data_subsets = []

        ###########number_of_classes definde by param
        ###########define model_name
        # self.number_of_classes = 100
        self.number_of_classes = number_of_classes
        self.model_name = model_name
        self.data_name = data_name
        self.adv_patch_cpu=adv_patch_cpu
        self.attack_train_backdoor_gradascent=None

        self.best_inference_acc_train=0
        self.best_inference_acc_test=0
        self.l2norm_member_round=[]
        self.l2norm_nonmember_round=[]
        self.loss_member_round=[]
        self.loss_nonmember_round=[]

        self.save_information_count=attack_data['save_information_count']
        self.backdoor_way=attack_data['backdoor_way']
        self.backdoor_epoch=attack_data['backdoor_epoch']
        self.maxdiff=attack_data['maxdiff']
        self.a_b_ab=attack_data['a_b_ab']
        
        self.save_model_patch=experiment_param['model_path']
        self.attack_patch_list=[]
        self.ratio=attack_data['ratio']
        self.activatedataset=None
        ############################################
        self.attack_train_data = None
        # define train_data_member and train_data_non_member
        self.attack_train_data_member = None
        self.attack_train_data_non_member = None

        self.attack_test_data = None
        self.target_model = None
        self.active_attack_optimizer = None
        self.active_learning_rate = attack_data['active_learning_rate']

        # Create Attack model based on the target model.
        if self.model_name == "AlexNet":
            if self.data_name == 'cifar10':
                self.attack_model = AttackModel(target_model=self.model,
                                                number_of_classes=self.number_of_classes,
                                                number_of_observed_models=1,
                                                kernel=self.number_of_classes,
                                                linearsize=256)
            else:
                self.attack_model = AttackModel(target_model=self.model,
                                                number_of_classes=self.number_of_classes,
                                                number_of_observed_models=1,
                                                kernel=self.number_of_classes,
                                                linearsize=256)
        elif self.model_name == "CNNCifar10":
            self.attack_model = AttackModel(target_model=self.model,
                                            number_of_classes=self.number_of_classes,
                                            number_of_observed_models=1,
                                            kernel=self.number_of_classes,
                                            linearsize=84)
        elif self.model_name == "CNNFashion_Mnist":
            self.attack_model = AttackModel(target_model=self.model,
                                            number_of_classes=self.number_of_classes,
                                            number_of_observed_models=1,
                                            kernel=self.number_of_classes,
                                            linearsize=7 * 7 * 32)
        elif self.model_name == "CNNMnist":
            self.attack_model = AttackModel(target_model=self.model,
                                            number_of_classes=self.number_of_classes,
                                            number_of_observed_models=1,
                                            kernel=self.number_of_classes,
                                            linearsize=50)
        elif self.model_name == "LeNet5":
            self.attack_model = AttackModel(target_model=self.model,
                                            number_of_classes=self.number_of_classes,
                                            number_of_observed_models=1,
                                            kernel=self.number_of_classes,
                                            linearsize=84)
        elif self.model_name=="CNNModel":
            self.attack_model = AttackModel(target_model=self.model,
                                            number_of_classes=self.number_of_classes,
                                            number_of_observed_models=1,
                                            kernel=self.number_of_classes,
                                            linearsize=256)

        # Set attack optimizer after sending model to device
        self.attack_optimizer = optimizers.__dict__[self.attack_optimizer_name](
            params=self.attack_model.parameters(),
            lr=self.attack_model_learning_rate
        )

        #################################################

        self.is_poison = backdoor_data['is_poison']
        self.back_rounds = backdoor_data['back_rounds']
        self.back_prop = backdoor_data['back_prop']
        self.test_back_prop=backdoor_data['test_back_prop']
        self.noise_percentage = backdoor_data['noise_percentage']
        self.probability_threshold = backdoor_data['probability_threshold']
        self.max_iteration = backdoor_data['max_iteration']
        self.target = backdoor_data['target']
        self.patch_lr = backdoor_data['patch_lr']
        self.patch_type = backdoor_data['patch_type']
        self.log_dir = backdoor_data['log_dir']
        self.use_mialoss_ratio=backdoor_data['use_mialoss_ratio']
        self.mia_back_rounds = backdoor_data['mia_back_rounds']
        #################################################

        message = f"[ Attacker Settings: \n" \
                  f"| Attacker Stable In : {attack_data['stable_in']} \n" \
                  f"| Attacker Normal Training : {attack_data['normal_training']} \n" \
                  f"| Attack Gradient Ascent : {attack_data['is_gradient_ascent']} \n" \
                  f"| Attack Data Distribution: {self.attack_data_distribution} \n" \
                  f"| Attack LR: {self.attack_model_learning_rate} \n" \
                  f"| Attack Train Batch: {self.attack_train_batch_size} \n" \
                  f"| Attack Test Batch: {self.attack_test_batch_size} \n" \
                  f"| Attack optim: {self.attack_optimizer_name} \n" \
                  f"| Attack Local Epoch: {attack_data['attack_local_epoch']} \n" \
                  f"| Backdoor Local Epoch: {attack_data['backdoor_epoch']} \n" \
                  f"| Attack Loss: {attack_data['attack_loss_function']} \n" \
                  f"| Attack Ratio: {attack_data['ratio']} \n" \
                  f"| Active LR: {self.active_learning_rate} ]\n" \
                  f"[ Backdoor Settings: \n" \
                  f"| Backdoor Gradient Ascent: {backdoor_data['is_sga']} \n" \
                  f"| Backdoor Poison: {self.is_poison} \n" \
                  f"| Backdoor Train Patch: {backdoor_data['is_train']} \n" \
                  f"| Backdoor Back Rounds: {self.back_rounds} \n" \
                  f"| Backdoor Back Rounds: {self.mia_back_rounds} \n" \
                  f"| Backdoor Back Probability: {self.back_prop} \n" \
                  f"| Backdoor Test Back Probability: {self.test_back_prop} \n" \
                  f"| Backdoor Noise Percentage: {self.noise_percentage} \n" \
                  f"| Backdoor Probability Threshold: {self.probability_threshold} \n" \
                  f"| Backdoor Max Iteration: {self.max_iteration} \n" \
                  f"| Backdoor Target Label: {self.target} \n" \
                  f"| Backdoor Patch LR: {self.patch_lr} \n" \
                  f"| Backdoor Patch Type: {self.patch_type}]\n"
                  #f"| Backdoor Patch Type: {self.patch_type} ]\n"
        logging.info(msg=message)
        log_attack = logging.getLogger('log_attack')
        log_attack.info(message)


    def load_attack_data(self, train_data, test_data, client_indices, client_dataset_size,rootpath,poison_dataset):
        '''
        ###########################
        # 當類別過少時,會出現每個類的成員和非成員數據不夠取的情況，需要增加client_indices的長度
        ###########################
        """ Ja hier moet dus documentatie """
        logging.debug(' Loading datasets for attacker')
        message=' Loading datasets for attacker'
        log_attack = logging.getLogger('log_attack')
        log_attack.info(message)
        # Get distributions for attack data from server.
        distribution = self.attack_data_distribution

        # print("len(client_indices)",len(client_indices)) #50000
        # print("client_dataset_size",client_dataset_size) #12500

        # check is data sizes are correct:
        for length in distribution:#length need >=100 and %100==0
            assert length // 100 != 0 and length % 100 == 0
        assert len(client_indices) >= client_dataset_size
        assert len(client_indices) >= distribution[0] + distribution[1]

        # num of member and non member for train attack need increase
        # train_member_length = distribution[0] // 100
        # train_non_member_length = distribution[1] // 100
        # test_member_length = distribution[2] // 100
        # test_non_member_length = distribution[3] // 100
        train_member_length = distribution[0] // self.number_of_classes
        train_non_member_length = distribution[1] // self.number_of_classes
        test_member_length = distribution[2] // self.number_of_classes
        test_non_member_length = distribution[3] // self.number_of_classes
        #print("train_member_length",train_member_length) #20

        # Set index lists to empty list.
        training_member_index, training_non_member_index, test_member_index, test_non_member_index =  [], [], [],[]

        ##################################sort dataset by label
        train_data_sort_indices = sorted(range(len(train_data)), key=lambda i: train_data[i][1])  # 按标签排序的索引
        train_data_sort = Subset(train_data, train_data_sort_indices)

        test_data_sort_indices = sorted(range(len(test_data)), key=lambda i: test_data[i][1])  # 按标签排序的索引
        test_data_sort = Subset(test_data, test_data_sort_indices)

        train_class_num = len(train_data) // self.number_of_classes
        test_class_num = len(test_data) // self.number_of_classes
        ##################################

        # Get the indicis for the right amount of training and test samples.
        # may want to take same number from each class
        for target in range(self.number_of_classes):
            # Set the boundaries for picking samples from the right classes
            low_member_train, high_member_train = target * train_class_num, (target + 1) * train_class_num
            low_non_member_train, high_non_member_train = target * train_class_num, (target + 1) * train_class_num

            low_member_test, high_member_test = target * test_class_num, (target + 1) * test_class_num
            low_non_member_test, high_non_member_test = target * test_class_num, (target + 1) * test_class_num
            low_train_backdoor_gradascent,high_train_backdoor_gradascent=target * train_class_num, (target + 1) * train_class_num
            random_subset = list([x for x in client_indices if low_member_train <= x <= high_member_train])
            random.shuffle(random_subset)


            for _ in range(train_member_length):
                training_member_index.append(random_subset.pop())

            for _ in range(test_member_length):
                test_member_index.append(random_subset.pop())

            # random_subset = list(range(low_non_member, high_non_member))
            ##########################
            #random_subset = list(
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
        self.attack_train_data = (train_member, train_non_member)
        # attack_train_data_member and attack_train_non_member get value
        self.attack_train_data_member = train_member
        self.attack_train_non_member = train_non_member
        self.attack_test_data = (test_member, test_non_member)

        print("test_non_member",len(test_non_member)) #2000
        print("test_member", len(test_member)) #2000
        print("train_non_member", len(train_non_member))  # 2000
        print("train_member", len(train_member))  # 2000

        logging.debug('loaded attacker data successfully')
        message='loaded attacker data successfully'
        log_attack = logging.getLogger('log_attack')
        log_attack.info(message)
        '''


        distribution = self.attack_data_distribution
        train_member_length = distribution[0]
        train_non_member_length = distribution[1]
        test_member_length = distribution[2]
        test_non_member_length = distribution[3]
        if self.data_name=='cifar10':
            attack_train_member_path = './cifar10data/attack_train_member_' + str(train_member_length) + '.pth'
            attack_train_nonmember_path = './cifar10data/attack_train_nonmember_' + str(
                train_non_member_length) + '.pth'
            attack_test_member_path = './cifar10data/attack_test_member_' + str(test_member_length) + '.pth'
            attack_test_nonmember_path = './cifar10data/attack_test_nonmember_' + str(test_non_member_length) + '.pth'
        elif self.data_name=='mnist':
            attack_train_member_path = './mnistdata/attack_train_member_' + str(train_member_length) + '.pth'
            attack_train_nonmember_path = './mnistdata/attack_train_nonmember_' + str(
                train_non_member_length) + '.pth'
            attack_test_member_path = './mnistdata/attack_test_member_' + str(test_member_length) + '.pth'
            attack_test_nonmember_path = './mnistdata/attack_test_nonmember_' + str(test_non_member_length) + '.pth'
        elif self.data_name == 'fashion_mnist':
            attack_train_member_path = './fashion_mnistdata/attack_train_member_' + str(train_member_length) + '.pth'
            attack_train_nonmember_path = './fashion_mnistdata/attack_train_nonmember_' + str(
                train_non_member_length) + '.pth'
            attack_test_member_path = './fashion_mnistdata/attack_test_member_' + str(test_member_length) + '.pth'
            attack_test_nonmember_path = './fashion_mnistdata/attack_test_nonmember_' + str(test_non_member_length) + '.pth'
        else:
            attack_train_member_path = './cifar100data/attack_train_member_' + str(train_member_length) + '.pth'
            attack_train_nonmember_path = './cifar100data/attack_train_nonmember_' + str(train_non_member_length) + '.pth'
            attack_test_member_path = './cifar100data/attack_test_member_' + str(test_member_length) + '.pth'
            attack_test_nonmember_path = './cifar100data/attack_test_nonmember_' + str(test_non_member_length) + '.pth'

        
        train_member = torch.load(attack_train_member_path)
        train_non_member = torch.load(attack_train_nonmember_path)
        test_member = torch.load(attack_test_member_path)
        test_non_member = torch.load(attack_test_nonmember_path)
        if poison_dataset==1:
            '''
            trigger_path=rootpath+'/trigger/'
            folder_path=trigger_path
            # 获取文件夹中以.npy结尾的所有文件
            npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

            #   从文件名中提取触发值（trigger）作为排序的关键字
            trigger_values = [int(file_name.split('_')[0]) for file_name in npy_files]

            # 找到最大的触发值
            max_trigger_value = max(trigger_values)

            # 构造最大触发值对应的文件名
            max_file_name = f"{max_trigger_value}_trigger.npy"

            # 拼接完整的文件路径
            max_file_path = os.path.join(folder_path, max_file_name)


            print(f"The file with the maximum trigger value is: {max_file_path}")

            self.adv_patch_cpu=np.load(max_file_path)
            '''
            if self.data_name == 'mnist' or self.data_name == 'fashion_mnist':
                #train_member = MyDataset_mnist(self.adv_patch_cpu, train_member, self.target, self.test_back_prop, mode="train")
                #train_non_member = MyDataset_mnist(self.adv_patch_cpu, train_non_member, self.target, self.test_back_prop, mode="train")
                #test_member = MyDataset_mnist(self.adv_patch_cpu, test_member, self.target, self.test_back_prop, mode="train")
                #test_non_member = MyDataset_mnist(self.adv_patch_cpu, test_non_member, self.target, self.test_back_prop, mode="train")
                train_member_poison = MyDataset_mnist(self.adv_patch_cpu, train_member,
                                                    self.target, self.back_prop, mode="train")
                
                train_non_member_poison=MyDataset_mnist(self.adv_patch_cpu, train_non_member,
                                                    self.target, self.back_prop, mode="train")

                test_member_poison=MyDataset_mnist(self.adv_patch_cpu, test_member,
                                                    self.target, self.back_prop, mode="train")

                test_non_member_poison=MyDataset_mnist(self.adv_patch_cpu, test_non_member,
                                                    self.target, self.back_prop, mode="train")



            else:
                train_member_poison = MyDataset_cifar_new(self.adv_patch_cpu, train_member,
                                                    self.target, self.back_prop, mode="train",
                                                    number_of_classes=self.number_of_classes)
                train_non_member_poison=MyDataset_cifar_new(self.adv_patch_cpu, train_non_member,
                                                    self.target, self.back_prop, mode="train",
                                                    number_of_classes=self.number_of_classes)

                test_member_poison=MyDataset_cifar_new(self.adv_patch_cpu, test_member,
                                                    self.target, self.back_prop, mode="train",
                                                    number_of_classes=self.number_of_classes)
                test_non_member_poison=MyDataset_cifar_new(self.adv_patch_cpu, test_non_member,
                                                    self.target, self.back_prop, mode="train",
                                                   number_of_classes=self.number_of_classes)

        
            self.attack_train_data_member_poison = train_member_poison
            self.attack_train_non_member_poison = train_non_member_poison

            self.attack_test_member_poison=test_member_poison
            self.attack_test_non_member_poison=test_non_member_poison


        self.attack_train_data = (train_member, train_non_member)
        # attack_train_data_member and attack_train_non_member get value
        self.attack_train_data_member = train_member
        self.attack_train_non_member = train_non_member
        self.attack_test_data = (test_member, test_non_member)

        print("test_non_member", len(test_non_member))  # 2000
        print("test_member", len(test_member))  # 2000
        print("train_non_member", len(train_non_member))  # 2000
        print("train_member", len(train_member))  # 2000

        logging.debug('loaded attacker data successfully')
        message = 'loaded attacker data successfully'
        log_attack = logging.getLogger('log_attack')
        log_attack.info(message)

        print("train_member", len(train_member))  # 2000
        if self.data_name=='cifar-100-python':
            temp = []
            for i in range(100):
                temp.append(0)
        else:
            temp = []
            for i in range(10):
                temp.append(0)

        for item in train_member:
            temp[item[1]] += 1
        print(temp)

        print("test_member", len(test_member))  # 2000
        if self.data_name=='cifar-100-python':
            temp = []
            for i in range(100):
                temp.append(0)
        else:
            temp = []
            for i in range(10):
                temp.append(0)

        for item in test_member:
            temp[item[1]] += 1
        print(temp)

        print("train_non_member", len(train_non_member))  # 2000
        if self.data_name=='cifar-100-python':
            temp = []
            for i in range(100):
                temp.append(0)
        else:
            temp = []
            for i in range(10):
                temp.append(0)

        for item in train_non_member:
            temp[item[1]] += 1
        print(temp)

        print("test_non_member", len(test_non_member))  # 2000
        if self.data_name=='cifar-100-python':
            temp = []
            for i in range(100):
                temp.append(0)
        else:
            temp = []
            for i in range(10):
                temp.append(0)

        for item in test_non_member:
            temp[item[1]] += 1
        print(temp)



    def genactivatedataloader(self,clientdataset):
        datalen=int(self.ratio*len(clientdataset))
        subset_indices = random.sample(range(len(clientdataset)), datalen)

        # 创建子集数据集
        subset_dataset = Subset(clientdataset, subset_indices)
        self.activatedataset=subset_dataset

        print("local attack dataset", len(self.activatedataset))  # 2000
        if self.data_name=='cifar-100-python':
            temp = []
            for i in range(100):
                temp.append(0)
        else:
            temp = []
            for i in range(10):
                temp.append(0)

        for item in self.activatedataset:
            temp[item[1]] += 1
        print(temp)
        







    def save_patch(self,path,numlist):
        num=0
        for patch in self.attack_patch_list:
            savepath=path+'/'+str(numlist[num])+'_trigger.npy'
            print(savepath)
            np.save(savepath,patch)
            print(patch)
            num+=1

    def chose_max_epoch(self,target_models,ori_models,save_img_path):
        loss_diff_list=[]
        l2norm_diff_list=[]
        loss_diff_list_ori=[]
        l2norm_diff_list_ori=[]

        loss_tar_all_list,loss_ori_all_list =[],[]
        correct_tar_all_list,correct_ori_all_list = [],[]
        l2norm_tar_all_list,l2norm_ori_all_list=[],[]

        losses_tar_member_list,losses_ori_member_list = [],[]
        correct_tar_member_list,correct_ori_member_list = [],[]

        losses_tar_nonmember_list,losses_ori_nonmember_list = [],[]
        correct_tar_nonmember_list,correct_ori_nonmember_list = [],[]

        l2norm_tar_member_list,l2norm_ori_member_list = [],[]
        l2norm_tar_nonmember_list,l2norm_ori_nonmember_list = [],[]






        member, non_member = self.attack_test_data

        seed = 1
        with open('./config.yaml') as c:
            configs = list(yaml.load_all(c, Loader=yaml.FullLoader))
        log_config = configs[4]["LOG_settings"]
        with open('./log/persistent.txt', 'r', encoding="utf-8") as f:
            read_data = int(f.read())
            file_number = str(read_data)

        # create date string for unique file_naming
        date_string = log_config['log_path'] \
                        + str(datetime.now().month) + '_' \
                        + str(datetime.now().day) + '/'

        log_filename_chose = date_string + file_number + '_Chose_' + str(seed) + '.txt'



        log_chose = logging.getLogger('log_chose')
        file_handle_chose = logging.FileHandler(log_filename_chose)
        log_chose.addHandler(file_handle_chose)

        member_data_loader = DataLoader(member,
                                        batch_size=1,
                                        shuffle=True,
                                        num_workers=0,
                                        pin_memory=True
                                        )

        nonmember_data_loader = DataLoader(non_member,
                                           batch_size=1,
                                           shuffle=True,
                                           num_workers=0,
                                           pin_memory=True
                                           )



        count = 0


        for model in tqdm(target_models):
            count += 1
            l2norm_member_bacth = []
            l2norm_nonmember_bacth = []
            loss_member_batch = []
            loss_nonmember_batch = []
            model = model.to(self.device)
            model.eval()

            losses_tar_all = AverageMeter()
            correct_tar_all = 0
            l2norm_tar_all=AverageMeter()

            losses_tar_member = AverageMeter()
            correct_tar_member = 0

            losses_tar_nonmember = AverageMeter()
            correct_tar_nonmember = 0

            l2norm_tar_member = AverageMeter()
            l2norm_tar_nonmember = AverageMeter()



            for member_input, member_target in member_data_loader:
                model.zero_grad()
                member_input, member_target = member_input.float().to(self.device), member_target.long().to(self.device)

                outputs = model(member_input)
                loss = self.loss_function(outputs, member_target)

                loss_member_batch.append(loss.item())
                losses_tar_all.update(loss.item())
                losses_tar_member.update(loss.item())

                _, predicted = torch.max(outputs.data, 1)
                correct_tar_all += (predicted == member_target).sum().item()
                correct_tar_member+=(predicted == member_target).sum().item()

                loss.backward()
                if self.model_name == "AlexNet":
                    current_grads = model.classifier.weight.grad
                elif self.model_name == "CNNCifar10":
                    current_grads = model.fc3.weight.grad
                elif self.model_name == "CNNFashion_Mnist":
                    current_grads = model.fc.weight.grad
                elif self.model_name == "CNNMnist":
                    current_grads = model.fc2.weight.grad
                elif self.model_name == "LeNet5":
                    current_grads = model.fc3.weight.grad
                # current_grads = model.lastlayer_name.weight.grad.view([1, 1, 256, self.number_of_classes])
                elif self.model_name == "CNNModel":
                    current_grads = model.fc2.weight.grad
                else:
                    current_grads = None
                l2_norm_member = torch.norm(current_grads, p=2)
                l2norm_member_bacth.append(l2_norm_member.item())
                l2norm_tar_member.update(l2_norm_member.item())
                l2norm_tar_all.update(l2_norm_member.item())


            for nonmember_input, nonmember_target in nonmember_data_loader:
                model.zero_grad()
                nonmember_input, nonmember_target = nonmember_input.float().to(self.device), nonmember_target.long().to(
                    self.device)

                outputs = model(nonmember_input)
                loss = self.loss_function(outputs, nonmember_target)
                loss_nonmember_batch.append(loss.item())

                losses_tar_all.update(loss.item())
                losses_tar_nonmember.update(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                correct_tar_all += (predicted == nonmember_target).sum().item()
                correct_tar_nonmember += (predicted == nonmember_target).sum().item()



                loss.backward()
                if self.model_name == "AlexNet":
                    current_grads = model.classifier.weight.grad
                elif self.model_name == "CNNCifar10":
                    current_grads = model.fc3.weight.grad
                elif self.model_name == "CNNFashion_Mnist":
                    current_grads = model.fc.weight.grad
                elif self.model_name == "CNNMnist":
                    current_grads = model.fc2.weight.grad
                elif self.model_name == "LeNet5":
                    current_grads = model.fc3.weight.grad
                # current_grads = model.lastlayer_name.weight.grad.view([1, 1, 256, self.number_of_classes])
                elif self.model_name == "CNNModel":
                    current_grads = model.fc2.weight.grad
                else:
                    current_grads = None
                l2_norm_nonmember = torch.norm(current_grads, p=2)

                l2norm_nonmember_bacth.append(l2_norm_nonmember.item())
                l2norm_tar_nonmember.update(l2_norm_nonmember.item())
                l2norm_tar_all.update(l2_norm_nonmember.item())

            model.to('cpu')
            plt.figure()
            l2_min = 0
            l2_max = int(max(l2norm_nonmember_bacth))

            bins = int(l2_max / 0.5)

            #print(l2_max)
            #print(bins)
            #print(l2norm_nonmember_bacth)
            #print(l2norm_member_bacth)
            counts_p, _, _ = plt.hist(x=l2norm_member_bacth, bins=bins, range=(l2_min, l2_max), density=True)
            counts_q, _, _ = plt.hist(x=l2norm_nonmember_bacth, bins=bins, range=(l2_min, l2_max), density=True)

            total_p = sum(counts_p)
            total_q = sum(counts_q)
            p = [count / total_p for count in counts_p]
            q = [count / total_q for count in counts_q]
            js_distance = distance.jensenshannon(p, q)
            js_distance_l2norm = round(js_distance, 5)

            loss_min = 0;
            loss_max = int(max(loss_nonmember_batch))
            bins = int(loss_max / 0.5)
            counts_p, _, _ = plt.hist(x=loss_member_batch, bins=bins, range=(loss_min, loss_max), density=False)
            counts_q, _, _ = plt.hist(x=loss_nonmember_batch, bins=bins, range=(loss_min, loss_max), density=False)

            total_p = sum(counts_p)
            total_q = sum(counts_q)
            p = [count / total_p for count in counts_p]
            q = [count / total_q for count in counts_q]
            js_distance = distance.jensenshannon(p, q)
            js_distance_loss = round(js_distance, 5)

            loss_diff_list.append(js_distance_loss)
            l2norm_diff_list .append(js_distance_l2norm)

            loss_tar_all_list.append(losses_tar_all.avg)
            correct_tar_all_list.append(correct_tar_all/(len(member)+len(non_member)))
            l2norm_tar_all_list.append(l2norm_tar_all.avg)

            losses_tar_member_list.append(losses_tar_member.avg)
            correct_tar_member_list.append(correct_tar_member/len(member))

            losses_tar_nonmember_list.append(losses_tar_nonmember.avg)
            correct_tar_nonmember_list.append(correct_tar_nonmember/len(non_member))

            l2norm_tar_member_list.append(l2norm_tar_member.avg)
            l2norm_tar_nonmember_list.append(l2norm_tar_nonmember.avg)





        for model in tqdm(ori_models):
            count += 1
            l2norm_member_bacth = []
            l2norm_nonmember_bacth = []
            loss_member_batch = []
            loss_nonmember_batch = []
            model = model.to(self.device)
            model.eval()

            losses_ori_all = AverageMeter()
            correct_ori_all = 0
            l2norm_ori_all=AverageMeter()

            losses_ori_member = AverageMeter()
            correct_ori_member = 0

            losses_ori_nonmember = AverageMeter()
            correct_ori_nonmember = 0

            l2norm_ori_member = AverageMeter()
            l2norm_ori_nonmember = AverageMeter()

            for member_input, member_target in member_data_loader:
                model.zero_grad()
                member_input, member_target = member_input.float().to(self.device), member_target.long().to(self.device)

                outputs = model(member_input)
                loss = self.loss_function(outputs, member_target)

                loss_member_batch.append(loss.item())
                losses_ori_all.update(loss.item())
                losses_ori_member.update(loss.item())

                _, predicted = torch.max(outputs.data, 1)
                correct_ori_all += (predicted == member_target).sum().item()
                correct_ori_member += (predicted == member_target).sum().item()



                loss.backward()
                if self.model_name == "AlexNet":
                    current_grads = model.classifier.weight.grad
                elif self.model_name == "CNNCifar10":
                    current_grads = model.fc3.weight.grad
                elif self.model_name == "CNNFashion_Mnist":
                    current_grads = model.fc.weight.grad
                elif self.model_name == "CNNMnist":
                    current_grads = model.fc2.weight.grad
                elif self.model_name == "LeNet5":
                    current_grads = model.fc3.weight.grad
                # current_grads = model.lastlayer_name.weight.grad.view([1, 1, 256, self.number_of_classes])
                elif self.model_name == "CNNModel":
                    current_grads = model.fc2.weight.grad
                else:
                    current_grads = None
                l2_norm_member = torch.norm(current_grads, p=2)
                l2norm_member_bacth.append(l2_norm_member.item())
                l2norm_ori_member.update(l2_norm_member.item())
                l2norm_ori_all.update(l2_norm_member.item())

            for nonmember_input, nonmember_target in nonmember_data_loader:
                model.zero_grad()
                nonmember_input, nonmember_target = nonmember_input.float().to(self.device), nonmember_target.long().to(
                    self.device)

                outputs = model(nonmember_input)
                loss = self.loss_function(outputs, nonmember_target)

                loss_nonmember_batch.append(loss.item())

                losses_ori_all.update(loss.item())
                losses_ori_nonmember.update(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                correct_ori_all += (predicted == nonmember_target).sum().item()
                correct_ori_nonmember += (predicted == nonmember_target).sum().item()


                loss.backward()
                if self.model_name == "AlexNet":
                    current_grads = model.classifier.weight.grad
                elif self.model_name == "CNNCifar10":
                    current_grads = model.fc3.weight.grad
                elif self.model_name == "CNNFashion_Mnist":
                    current_grads = model.fc.weight.grad
                elif self.model_name == "CNNMnist":
                    current_grads = model.fc2.weight.grad
                elif self.model_name == "LeNet5":
                    current_grads = model.fc3.weight.grad
                elif self.model_name == "CNNModel":
                    current_grads = model.fc2.weight.grad
                # current_grads = model.lastlayer_name.weight.grad.view([1, 1, 256, self.number_of_classes])
                else:
                    current_grads = None
                l2_norm_nonmember = torch.norm(current_grads, p=2)

                l2norm_nonmember_bacth.append(l2_norm_nonmember.item())
                l2norm_ori_nonmember.update(l2_norm_nonmember.item())
                l2norm_ori_all.update(l2_norm_nonmember.item())


            model.to('cpu')
            plt.figure()
            l2_min = 0
            l2_max = int(max(l2norm_nonmember_bacth))


            bins = int(l2_max / 0.02)

            counts_p, _, _ = plt.hist(x=l2norm_member_bacth, bins=bins, range=(l2_min, l2_max), density=True)
            counts_q, _, _ = plt.hist(x=l2norm_nonmember_bacth, bins=bins, range=(l2_min, l2_max), density=True)

            total_p = sum(counts_p)
            total_q = sum(counts_q)
            p = [count / total_p for count in counts_p]
            q = [count / total_q for count in counts_q]
            js_distance = distance.jensenshannon(p, q)
            js_distance_l2norm = round(js_distance, 5)

            loss_min = 0;
            loss_max = int(max(loss_nonmember_batch))
            bins = int(loss_max / 0.2)
            counts_p, _, _ = plt.hist(x=loss_member_batch, bins=bins, range=(loss_min, loss_max), density=False)
            counts_q, _, _ = plt.hist(x=loss_nonmember_batch, bins=bins, range=(loss_min, loss_max), density=False)

            total_p = sum(counts_p)
            total_q = sum(counts_q)
            p = [count / total_p for count in counts_p]
            q = [count / total_q for count in counts_q]
            js_distance = distance.jensenshannon(p, q)
            js_distance_loss = round(js_distance, 5)

            loss_diff_list_ori.append(js_distance_loss)
            l2norm_diff_list_ori.append(js_distance_l2norm)

            loss_ori_all_list.append(losses_ori_all.avg)
            correct_ori_all_list.append(correct_ori_all/(len(member)+len(non_member)))
            l2norm_ori_all_list.append(l2norm_ori_all.avg)

            losses_ori_member_list.append(losses_ori_member.avg)
            correct_ori_member_list.append(correct_ori_member/len(member))

            losses_ori_nonmember_list.append(losses_ori_nonmember.avg)
            correct_ori_nonmember_list.append(correct_ori_nonmember/len(non_member))

            l2norm_ori_member_list.append(l2norm_ori_member.avg)
            l2norm_ori_nonmember_list.append(l2norm_ori_nonmember.avg)



        all_targer_list=[a+b for a,b in zip(loss_diff_list,l2norm_diff_list)]
        all_ori_list=[a+b for a,b in zip(loss_diff_list_ori,l2norm_diff_list_ori)]

        chazhi_loss_diff_list=[x - y for x,y in zip(loss_diff_list,loss_diff_list_ori)]
        chazhi_l2norm_diff_list=[x - y for x,y in zip(l2norm_diff_list,l2norm_diff_list_ori)]
        chazhi_all_diff_list=[x - y for x,y in zip(all_targer_list,all_ori_list)]



        # 使用enumerate函数创建带有索引的元组列表
        indexed_list = list(enumerate(chazhi_all_diff_list))

        # 使用sorted函数对列表进行排序，按元素的值排序
        sorted_list_all = sorted(indexed_list, key=lambda x: x[1])

        # 输出排序后的元素和其原始索引
        start, end, jiange = self.maxdiff
        message = f'L2-Norm+Loss Sort Res '

        log_chose = logging.getLogger('log_chose')
        log_chose.info(message)

        for index, value in sorted_list_all[-5:]:
            message = f"L2-Norm+Loss 原始索引：{start + index * jiange}, 值：{value}"
            #print(f"L2-Norm+Loss 原始索引：{start + index * jiange}, 值：{value}")
            log_chose = logging.getLogger('log_chose')
            log_chose.info(message)

        for index, value in sorted_list_all:
            print(f"L2-Norm+Loss 原始索引：{start + index * jiange}, 值：{value}")



        # 使用enumerate函数创建带有索引的元组列表
        indexed_list = list(enumerate(chazhi_l2norm_diff_list))

        # 使用sorted函数对列表进行排序，按元素的值排序
        sorted_list_l2norm = sorted(indexed_list, key=lambda x: x[1])

        # 输出排序后的元素和其原始索引
        start,end,jiange=self.maxdiff
        message = f'L2-Norm Sort Res '


        log_chose = logging.getLogger('log_chose')
        log_chose.info(message)


        for index, value in sorted_list_l2norm[-5:]:
            message = f"l2norm gs diff 原始索引：{start+index*jiange}, 值：{value}"
            #print(f"l2norm gs diff 原始索引：{start+index*jiange}, 值：{value}")
            log_chose = logging.getLogger('log_chose')
            log_chose.info(message)

        for index, value in sorted_list_l2norm:
            print(f"l2norm gs diff 原始索引：{start+index*jiange}, 值：{value}")


        # 使用enumerate函数创建带有索引的元组列表
        indexed_list = list(enumerate(chazhi_loss_diff_list))

        # 使用sorted函数对列表进行排序，按元素的值排序
        sorted_list_loss = sorted(indexed_list, key=lambda x: x[1])
        # 输出排序后的元素和其原始索引
        message = f'Loss Sort Res '
        log_chose = logging.getLogger('log_chose')
        log_chose.info(message)

        for index, value in sorted_list_loss[-5:]:
            message=f"loss gs diff 原始索引：{start+index*jiange}, 值：{value}"
            #print(f"loss gs diff 原始索引：{start+index*jiange}, 值：{value}")
            log_chose = logging.getLogger('log_chose')
            log_chose.info(message)


        for index, value in sorted_list_loss:
            print(f"loss gs diff 原始索引：{start+index*jiange}, 值：{value}")


        x = [i for i in range(start, end+1, jiange)]

        plt.figure()
        plt.plot(x, chazhi_loss_diff_list, label='loss_gs-round')
        plt.xlabel('round')
        plt.ylabel('loss_diff_gs')
        plt.xticks(fontsize=6,rotation=90)
        x_jiange=MultipleLocator(jiange)
        ax=plt.gca()
        ax.xaxis.set_major_locator(x_jiange)
        with open('./log/persistent.txt', 'r', encoding="utf-8") as f:
            read_data = int(f.read())
            file_number = str(read_data + 1)

        name = 'loss_gsdiff_round-' + file_number
        plt.title(name)
        name = save_img_path + name  + '.png'

        plt.legend()  # 添加图例
        plt.savefig(name)

        plt.figure()
        plt.plot(x, chazhi_l2norm_diff_list, label='l2norm_gs-round')
        plt.xlabel('round')
        plt.ylabel('l2norm_gs')
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_jiange)
        plt.xticks(fontsize=6, rotation=90)
        with open('./log/persistent.txt', 'r', encoding="utf-8") as f:
            read_data = int(f.read())
            #file_number = str(read_data + 1)

        name = 'l2norm_gs_round-' + file_number
        plt.title(name)
        name = save_img_path + name + '.png'

        plt.legend()  # 添加图例
        plt.savefig(name)

        plt.figure()
        plt.plot(x, chazhi_all_diff_list, label='l2norm+loss_gs-round')
        plt.xlabel('round')
        plt.ylabel('l2norm+loss_gs')
        x_jiange = MultipleLocator(jiange)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_jiange)
        plt.xticks(fontsize=6,rotation=90)
        with open('./log/persistent.txt', 'r', encoding="utf-8") as f:
            read_data = int(f.read())
            file_number = str(read_data + 1)

        name = 'l2norm+loss_gs-' + file_number
        plt.title(name)
        name = save_img_path + name + '.png'

        plt.legend()  # 添加图例
        plt.savefig(name)


        '''
        loss_tar_all_list,loss_ori_all_list =[],[]
        correct_tar_all_list,correct_ori_all_list = [],[]
        l2norm_tar_all_list,l2norm_ori_all_list=[],[]

        losses_tar_member_list,losses_ori_member_list = [],[]
        correct_tar_member_list,correct_ori_member_list = [],[]

        losses_tar_nonmember_list,losses_ori_nonmember_list = [],[]
        correct_tar_nonmember_list,correct_ori_nonmember_list = [],[]

        l2norm_tar_member_list,l2norm_ori_member_list = [],[]
        l2norm_tar_nonmember_list,l2norm_ori_nonmember_list = [],[]
        '''


        self.plot_diffimg(x,loss_tar_all_list,loss_ori_all_list,'round','loss','all-backdoor','all-gradascent','allloss_round',save_img_path)
        self.plot_diffimg(x,correct_tar_all_list,correct_ori_all_list,'round','correct','all-backdoor','all-gradascent','allcoreect_round',save_img_path)
        self.plot_diffimg(x,l2norm_tar_all_list,l2norm_ori_all_list,'round','l2norm','all-backdoor','all-gradascent','alll2norm_round',save_img_path)

        self.plot_diffimg(x,losses_tar_member_list,losses_ori_member_list,'round','loss','member-backdoor','member-gradascent','memberloss_round',save_img_path)
        self.plot_diffimg(x,correct_tar_member_list,correct_ori_member_list,'round','correct','member-backdoor','member-gradascent','membercorrect_round',save_img_path)

        self.plot_diffimg(x,losses_tar_nonmember_list,losses_ori_nonmember_list,'round','loss','nonmember-backdoor','nonmember-gradascent','nonmemberloss_round',save_img_path)
        self.plot_diffimg(x,correct_tar_nonmember_list,correct_ori_nonmember_list,'round','correct','nonmember-backdoor','nonmember-gradascent','nonmembercorrect_round',save_img_path)

        self.plot_diffimg(x, l2norm_tar_member_list, l2norm_ori_member_list, 'round', 'l2norm',
                          'member-backdoor', 'member-gradascent', 'memberl2norm_round',save_img_path)
        self.plot_diffimg(x, l2norm_tar_nonmember_list, l2norm_ori_nonmember_list, 'round', 'l2norm',
                          'nonmember-backdoor', 'nonmember-gradascent', 'nonmemberl2norm_round',save_img_path)

        self.plot_diffimg(x, l2norm_tar_member_list, l2norm_tar_nonmember_list, 'round', 'l2norm',
                          'member-backdoor', 'nonmember-backdoor', 'backdoorl2norm_round',save_img_path)

        self.plot_diffimg(x, losses_tar_member_list, losses_tar_nonmember_list, 'round', 'loss',
                          'member-backdoor', 'nonmember-backdoor', 'backdoorloss_round',save_img_path)



        self.plot_diffimg(x, losses_ori_member_list, losses_ori_nonmember_list, 'round', 'loss',
                          'member-gradascent', 'nonmember-gradascent', 'gradascentloss_round',save_img_path)

        self.plot_diffimg(x, l2norm_ori_member_list, l2norm_ori_nonmember_list, 'round', 'l2norm',
                          'member-gradascent', 'nonmember-gradascent', 'gradascentl2norm_round',save_img_path)

        ori_diff_membernomember_loss_list=[b-a for a,b in zip(losses_ori_member_list,losses_ori_nonmember_list)]
        tar_diff_membernomember_loss_list=[b-a for a,b in zip(losses_tar_member_list,losses_tar_nonmember_list)]
        self.plot_diffimg(x, tar_diff_membernomember_loss_list,ori_diff_membernomember_loss_list , 'round', 'loss-diff(non-mem)',
                          'backdoordiff', 'gradascentdiff', 'lossdiffbe_round',save_img_path)



        ori_diff_membernomember_l2norm_list = [b - a for a, b in zip(l2norm_ori_member_list, l2norm_ori_nonmember_list)]
        tar_diff_membernomember_l2norm_list = [b - a for a, b in zip(l2norm_tar_member_list, l2norm_tar_nonmember_list)]
        self.plot_diffimg(x, tar_diff_membernomember_l2norm_list, ori_diff_membernomember_l2norm_list, 'round', 'l2norm-diff(non-mem)',
                          'backdoordiff', 'gradascentdiff', 'l2normdiffbe_round',save_img_path)


        self.plot_diffimg(x, losses_ori_member_list, losses_ori_nonmember_list, 'round', 'loss',
                          'member-gradascent', 'nonmember-gradascent', 'gradascentloss_round',save_img_path)
        self.plot_diffimg(x, losses_ori_member_list, losses_ori_nonmember_list, 'round', 'loss',
                          'member-gradascent', 'nonmember-gradascent', 'gradascentloss_round',save_img_path)
        self.plot_diffimg(x, losses_ori_member_list, losses_ori_nonmember_list, 'round', 'loss',
                          'member-gradascent', 'nonmember-gradascent', 'gradascentloss_round',save_img_path)



    def plot_diffimg(self,x,y1,y2,xlabal,ylabal,labal1,labal2,figname,save_img_path):
        plt.figure()
        jiange=x[1]-x[0]
        plt.plot(x, y1, label=labal1,color='red')
        plt.plot(x, y2, label=labal2,color='blue')
        plt.xlabel(xlabal)
        plt.ylabel(ylabal)
        x_jiange = MultipleLocator(jiange)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_jiange)
        plt.xticks(fontsize=6,rotation=90)
        with open('./log/persistent.txt', 'r', encoding="utf-8") as f:
            read_data = int(f.read())
            file_number = str(read_data + 1)

        name = figname+'-' + file_number
        plt.title(name)
        name = save_img_path + name + '.png'

        plt.legend()  # 添加图例
        plt.savefig(name)


    def gradient_ascent_attack(self, round_number, attack_local_epoch):
        """ Ja hier moet dus documentatie """
        # Put model in training mode and load model onto device
        self.target_model = copy.deepcopy(self.model)

        ###################################################################################
        # optimizer

        self.active_attack_optimizer = optimizers.__dict__['Adam'](
            params=self.target_model.parameters(),
            # maximize=True,
            lr=self.active_learning_rate
        )
        '''
        self.active_attack_optimizer = optimizers.__dict__[self.optimizer_name](
            params=self.target_model.parameters(),
            lr=self.active_learning_rate,
            #lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        '''

        # self.optimizer = optimizers.__dict__[self.optimizer_name](
        #     params=self.target_model.parameters(),
        #     lr=self.learning_rate,
        #     momentum=self.momentum,
        #     weight_decay=self.weight_decay
        # )
        # self.active_attack_optimizer = self.optimizer
        ###################################################################################

        self.target_model.train()
        self.target_model.to(self.device)

        # Setup variables to log statistics.
        losses = 0
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        start_time = time.time()
        '''
        member, non_member = self.attack_train_data
        
        if self.a_b_ab==1:
            active_set = member
        elif self.a_b_ab==2:
            active_set = non_member
        else:
            active_set = torch.utils.data.ConcatDataset([member, non_member])
        '''
        member, non_member = self.attack_train_data        
        active_set = torch.utils.data.ConcatDataset([member, non_member])

        #active_set=self.activatedataset
        #active_set=non_member
        attack_data_loader = DataLoader(active_set,
                                        batch_size=4,
                                        shuffle=True,
                                        pin_memory=True
                                        )

        for epoch in range(attack_local_epoch):
            for data, labels in attack_data_loader:
                # Transfer data to CPU or GPU and set gradients to zero for performance.
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                self.active_attack_optimizer.zero_grad()

                # Do a forward pass through the network to get prediction values and update loss metric.
                outputs = self.target_model(data)
                loss = self.loss_function(outputs, labels)

                # Do a backward pass through the network to get the gradients
                # and then use the optimizer to update the weights.
                loss.backward()

                ####################################################
                for name, param in self.target_model.named_parameters():
                    param.grad *= -1
                    # print("", name, param)
                ###################################################

                self.active_attack_optimizer.step()
                loss_meter.update(loss.item())

        # Update loss and run_time metrics
        #loss_meter.update(losses)
        batch_time.update(time.time() - start_time)

        # message = f'[ Round: {round_number} ' \
        #           f'| Active Attack ' \
        #           f'| Client: {self.client_id} ' \
        #           f'| Time: {batch_time.avg:.2f}s ' \
        #           f'| Loss: {loss_meter.avg:.5f} '
        # logging.info(message)
        self.target_model.to("cpu")
        self.model = copy.deepcopy(self.target_model)


        ########################################################################
        self.model.eval()
        self.model.to(self.device)

        # load once to save time. To be used for calculating the accuracy.
        data_size = len(attack_data_loader.dataset)

        # Define attributes to keep track of scores during testing
        batch_time = AverageMeter()
        losses = AverageMeter()
        start_time = time.time()
        correct = 0

        with torch.no_grad():
            for data, labels in attack_data_loader:
                # Transfer data to CPU or GPU.
                data, labels = data.float().to(self.device), labels.long().to(self.device)

                # Do a forward pass through the network to get prediction values and update loss metric.
                outputs = self.model(data)
                # print(self.loss_function(outputs, labels))
                losses.update(self.loss_function(outputs, labels))

                # Compare predictions to labels and get accuracy score.
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                # Update time and accuracy metric.
                batch_time.update(time.time() - start_time)

            accuracy = (correct / data_size) * 100
            message = f'[ Round: {round_number} ' \
                      f'| Active Attack Eval ' \
                      f'| Client: {self.client_id} ' \
                      f'| Loss: {losses.avg:.5f} ' \
                      f'| Test_Acc ({correct}/{data_size})={accuracy:.2f}% ] '
            logging.info(message)


            log_attack = logging.getLogger('log_attack')
            log_attack.info(message)

        ########################################################################


    def other_training(self,round_number,local_epoch):
        # Put model in training mode and load model onto device
        self.target_model = copy.deepcopy(self.model)

        ###################################################################################
        # optimizer
        self.active_attack_optimizer = optimizers.__dict__['Adam'](
            params=self.target_model.parameters(),
            # maximize=True,
            lr=self.active_learning_rate
        )

        # self.optimizer = optimizers.__dict__[self.optimizer_name](
        #     params=self.target_model.parameters(),
        #     lr=self.learning_rate,
        #     momentum=self.momentum,
        #     weight_decay=self.weight_decay
        # )
        # self.active_attack_optimizer = self.optimizer
        ###################################################################################

        self.target_model.train()
        self.target_model.to(self.device)

        # Setup variables to log statistics.
        losses = 0
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        start_time = time.time()

        member, non_member = self.attack_train_data
        #active_set = torch.utils.data.ConcatDataset([member, non_member])
        active_set=member
        attack_data_loader = DataLoader(active_set,
                                        batch_size=self.attack_train_batch_size,
                                        shuffle=True,
                                        pin_memory=True
                                        )

        for epoch in range(local_epoch):
            for data, labels in attack_data_loader:
                # Transfer data to CPU or GPU and set gradients to zero for performance.
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                self.active_attack_optimizer.zero_grad()

                # Do a forward pass through the network to get prediction values and update loss metric.
                outputs = self.target_model(data)
                loss = self.loss_function(outputs, labels)

                # Do a backward pass through the network to get the gradients
                # and then use the optimizer to update the weights.
                loss.backward()

                ####################################################
                #for name, param in self.target_model.named_parameters():
                #    param.grad *= -1
                    # print("", name, param)
                ###################################################

                self.active_attack_optimizer.step()
                loss_meter.update(loss.item())

        # Update loss and run_time metrics
        # loss_meter.update(losses)
        batch_time.update(time.time() - start_time)

        # message = f'[ Round: {round_number} ' \
        #           f'| Active Attack ' \
        #           f'| Client: {self.client_id} ' \
        #           f'| Time: {batch_time.avg:.2f}s ' \
        #           f'| Loss: {loss_meter.avg:.5f} '
        # logging.info(message)
        self.target_model.to("cpu")
        self.model = copy.deepcopy(self.target_model)

        ########################################################################
        self.model.eval()
        self.model.to(self.device)

        # load once to save time. To be used for calculating the accuracy.
        data_size = len(attack_data_loader.dataset)

        # Define attributes to keep track of scores during testing
        batch_time = AverageMeter()
        losses = AverageMeter()
        start_time = time.time()
        correct = 0

        with torch.no_grad():
            for data, labels in attack_data_loader:
                # Transfer data to CPU or GPU.
                data, labels = data.float().to(self.device), labels.long().to(self.device)

                # Do a forward pass through the network to get prediction values and update loss metric.
                outputs = self.model(data)
                # print(self.loss_function(outputs, labels))
                losses.update(self.loss_function(outputs, labels))

                # Compare predictions to labels and get accuracy score.
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                # Update time and accuracy metric.
                batch_time.update(time.time() - start_time)

            accuracy = (correct / data_size) * 100
            message = f'[ Round: {round_number} ' \
                      f'| Active Attack Eval ' \
                      f'| Client: {self.client_id} ' \
                      f'| Loss: {losses.avg:.5f} ' \
                      f'| Test_Acc ({correct}/{data_size})={accuracy:.2f}% ] '
            logging.info(message)

            log_attack = logging.getLogger('log_attack')
            log_attack.info(message)

        ########################################################################


    def backdoor_attack_0(self, round_number,is_sga,is_train,image_size):
        """ Ja hier moet dus documentatie """
        # Put model in training mode and load model onto device
        self.target_model = copy.deepcopy(self.model)
        ###################################################################################
        # optimizer

        self.active_attack_optimizer = optimizers.__dict__['Adam'](
            params=self.target_model.parameters(),
            # maximize=True,
            lr=self.active_learning_rate
        )
        '''
        self.active_attack_optimizer = optimizers.__dict__[self.optimizer_name](
            params=self.target_model.parameters(),
            lr=self.active_learning_rate,
            #lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        '''
        ###################################################################################

        self.target_model.train()
        self.target_model.to(self.device)

        # Setup variables to log statistics.
        losses = 0
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        start_time = time.time()
        '''
        #member, non_member = self.attack_train_data
        member, non_member = self.attack_train_data
        if self.a_b_ab==1:
            active_set = member
        elif self.a_b_ab==2:
            active_set = non_member
        else:
            active_set = torch.utils.data.ConcatDataset([member, non_member])
        '''
        member, non_member = self.attack_train_data        
        active_set = torch.utils.data.ConcatDataset([member, non_member])
        #active_set=self.activatedataset
        #active_set=non_member
        #active_set=member
        ###################################################################################
        # train the patch
        temp_target_models_for_inference=[]
        temp_target_models_for_inference.append(self.target_model)
        if self.use_mialoss_ratio!=0 and is_train: attmodelres=self.get_attackmodel_loss(temp_target_models_for_inference).to(self.device)

        if is_train==1:
            attack_data_loader_backdooor = DataLoader(active_set,
                                            # dataset_poison,
                                            batch_size=1,
                                            shuffle=True,
                                            pin_memory=True
                                            )
            self.target_model=self.target_model.to(self.device)
            for i in range(self.back_rounds):
                # print("i",i)
                train_total, train_actual_total, train_success = 0, 0, 0
                for (image, label) in attack_data_loader_backdooor:
                    image, label = image.float().to(self.device), label.long().to(self.device)
                    #print("image", image.shape)  # torch.Size([1, 3, 32, 32])
                    #print("label", label) #tensor([87])

                    train_total += label.shape[0]
                    assert image.shape[0] == 1, 'Only one picture should be loaded each time.'

                    output= self.target_model(image)
                    _, predicted = torch.max(output.data, 1)

                    # print("predicted[0]",predicted[0]) #tensor(50)
                    # print("label[0]",label[0]) #tensor(87)
                    # print("predicted[0].data.cpu().numpy()",predicted[0].data.cpu().numpy()) #50
                    if predicted[0] != label[0] and predicted[0].data.cpu().numpy() != self.target:#target class
                        train_actual_total += 1

                        # 把随机生成的patch(3,3)放在空白图像（32，32）上
                        applied_patch, mask, x_location, y_location = mask_generation(self.patch_type, self.adv_patch_cpu, image_size=image_size)
                        #print("applied_patch",applied_patch.shape)  # (3, 32, 32)

                        # 把训练生成的patch(3,3)放在训练图像（32，32）上
                        
                        perturbated_image, applied_patch = self.patch_attack(image, applied_patch, mask, self.target, self.probability_threshold,
                                                                                self.target_model,self.use_mialoss_ratio,attmodelres, self.patch_lr, self.max_iteration)
                            
                            
                        # print("perturbated_image",perturbated_image.shape) # (1, 3, 32, 32)
                        # print("applied_patch",applied_patch.shape) # (1, 3, 32, 32)
                        # perturbated_images = perturbated_image.squeeze(0)
                        # plt.imshow(perturbated_images[0])
                        # plt.show()

                        perturbated_image = torch.from_numpy(perturbated_image)
                        # perturbated_image = torch.from_numpy(perturbated_image).cuda()
                        # print("perturbated_image",perturbated_image.shape) #torch.Size([1, 3, 32, 32])
                        perturbated_image=perturbated_image.to(self.device)

                        output= self.target_model(perturbated_image)
                        _, predicted = torch.max(output.data, 1)
                        if predicted[0].data.cpu().numpy() ==self.target:
                            train_success += 1

                        # 训练的patch
                        self.adv_patch_cpu = applied_patch[0][:, x_location:x_location + self.adv_patch_cpu.shape[1],
                                        y_location:y_location + self.adv_patch_cpu.shape[2]]
                        # print("self.adv_patch_cpu", self.adv_patch_cpu.shape)  # (3, 3, 3)
                        # plt.imshow(self.adv_patch_cpu[0])
                        # plt.show()
            # print("train_success", train_success)
        ###################################################################################

        ###################################################################################
        # print("train poison poison")
        # print("self.adv_patch_cpu_2",self.adv_patch_cpu)
        if self.data_name=='mnist' or self.data_name=='fashion_mnist':
            dataset_poison=MyDataset_mnist(self.adv_patch_cpu, active_set,self.target, self.back_prop, mode="train")
        else:
            dataset_poison = MyDataset_cifar_new(self.adv_patch_cpu, active_set,
                                                 self.target, self.back_prop, mode="train",number_of_classes=self.number_of_classes)
        attack_data_loader_backdoor = DataLoader(dataset_poison,
                                            batch_size=self.attack_train_batch_size,
                                            shuffle=True,
                                            pin_memory=True
                                            )
        self.attack_patch_list.append(self.adv_patch_cpu)
        # attack_data_loader_backdoor = DataLoader(active_set,
        #                                 batch_size=self.attack_train_batch_size,
        #                                 shuffle=True,
        #                                 pin_memory=True
        #                                 )
        ###################################################################################
        for epoch in range(self.backdoor_epoch):
        ###################################################################################
            for data, labels in attack_data_loader_backdoor:

                # Transfer data to CPU or GPU and set gradients to zero for performance.
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                self.active_attack_optimizer.zero_grad()

                # Do a forward pass through the network to get prediction values and update loss metric.
                outputs = self.target_model(data)
                loss = self.loss_function(outputs, labels)

                # Do a backward pass through the network to get the gradients
                # and then use the optimizer to update the weights.
                loss.backward()

                if is_sga>0:
                    #print("gradient ascent")
                    # torch.nn.utils.clip_grad_norm(self.target_model.parameters(), max_norm=1)
                    ####################################################
                    for name, param in self.target_model.named_parameters():
                        param.grad *= -1
                        # print("", name, param)
                    ###################################################

                self.active_attack_optimizer.step()
                losses += loss.item()
                loss_meter.update(loss.item())

        # Update loss and run_time metrics
        #loss_meter.update(losses)
        batch_time.update(time.time() - start_time)

        # message = f'[ Round: {round_number} ' \
        #           f'| Active Attack ' \
        #           f'| Client: {self.client_id} ' \
        #           f'| Time: {batch_time.avg:.2f}s ' \
        #           f'| Loss: {loss_meter.avg:.5f} '
        # logging.info(message)
        self.target_model.to("cpu")
        self.model = copy.deepcopy(self.target_model)


        ########################################################################
        self.model.eval()
        self.model.to(self.device)

        # load once to save time. To be used for calculating the accuracy.
        data_size = len(attack_data_loader_backdoor.dataset)

        # Define attributes to keep track of scores during testing
        batch_time = AverageMeter()
        losses = AverageMeter()
        start_time = time.time()
        correct = 0

        with torch.no_grad():
            for data, labels in attack_data_loader_backdoor:
                # Transfer data to CPU or GPU.
                data, labels = data.float().to(self.device), labels.long().to(self.device)

                # Do a forward pass through the network to get prediction values and update loss metric.
                outputs = self.model(data)
                # print(self.loss_function(outputs, labels))
                losses.update(self.loss_function(outputs, labels))

                # Compare predictions to labels and get accuracy score.
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                # Update time and accuracy metric.
                batch_time.update(time.time() - start_time)

            accuracy = (correct / data_size) * 100
            message = f'[ Round: {round_number} ' \
                      f'| Backdoor Attack Eval ' \
                      f'| Client: {self.client_id} ' \
                      f'| Loss: {losses.avg:.5f} ' \
                      f'| Backdoor Test_Acc ({correct}/{data_size})={accuracy:.2f}% ] '
            logging.info(message)
            log_attack = logging.getLogger('log_attack')
            log_attack.info(message)
        return self.adv_patch_cpu
        ########################################################################


    def backdoor_attack_1(self, round_number,is_sga,is_train,image_size):
        """ Ja hier moet dus documentatie """
        # Put model in training mode and load model onto device
        self.target_model = copy.deepcopy(self.model)

        ###################################################################################
        # optimizer
        self.active_attack_optimizer = optimizers.__dict__['Adam'](
            params=self.target_model.parameters(),
            # maximize=True,
            lr=self.active_learning_rate
        )
        '''
        self.active_attack_optimizer = optimizers.__dict__[self.optimizer_name](
            params=self.target_model.parameters(),
            lr=self.active_learning_rate,
            #lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        '''
        ###################################################################################

        self.target_model.train()
        self.target_model.to(self.device)

        # Setup variables to log statistics.
        losses = 0
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        start_time = time.time()
        '''
        #member, non_member = self.attack_train_data
        member, non_member = self.attack_train_data
        if self.a_b_ab == 1:
            active_set = member
        elif self.a_b_ab == 2:
            active_set = non_member
        else:
            active_set = torch.utils.data.ConcatDataset([member, non_member])
        '''
        member, non_member = self.attack_train_data        
        active_set = torch.utils.data.ConcatDataset([member, non_member])
        #active_set=self.activatedataset        
        temp_target_models_for_inference=[]
        temp_target_models_for_inference.append(self.target_model)
        if self.use_mialoss_ratio!=0 and is_train: attmodelres=self.get_attackmodel_loss(temp_target_models_for_inference).to(self.device)
        
        for epoch in range(self.backdoor_epoch):
            ###################################################################################
            # train the patch
            if is_train==1:
                attack_data_loader_backdooor = DataLoader(active_set,
                                                # dataset_poison,
                                                batch_size=1,
                                                shuffle=True,
                                                pin_memory=True
                                                )
                self.target_model=self.target_model.to(self.device)
                for i in range(self.back_rounds):
                    # print("i",i)
                    train_total, train_actual_total, train_success = 0, 0, 0
                    for (image, label) in attack_data_loader_backdooor:
                        image, label = image.float().to(self.device), label.long().to(self.device)
                        #print("image", image.shape)  # torch.Size([1, 3, 32, 32])
                        #print("label", label) #tensor([87])

                        train_total += label.shape[0]
                        assert image.shape[0] == 1, 'Only one picture should be loaded each time.'

                        output= self.target_model(image)
                        _, predicted = torch.max(output.data, 1)

                        # print("predicted[0]",predicted[0]) #tensor(50)
                        # print("label[0]",label[0]) #tensor(87)
                        # print("predicted[0].data.cpu().numpy()",predicted[0].data.cpu().numpy()) #50
                        if predicted[0] != label[0] and predicted[0].data.cpu().numpy() != self.target:#target class
                            train_actual_total += 1

                            # 把随机生成的patch(3,3)放在空白图像（32，32）上
                            applied_patch, mask, x_location, y_location = mask_generation(self.patch_type, self.adv_patch_cpu, image_size=image_size)
                            #print("applied_patch",applied_patch.shape)  # (3, 32, 32)


                            # 把训练生成的patch(3,3)放在训练图像（32，32）上
                            perturbated_image, applied_patch = self.patch_attack(image, applied_patch, mask, self.target, self.probability_threshold,
                                                                                self.target_model,self.use_mialoss_ratio,attmodelres, self.patch_lr, self.max_iteration)
                            
                            
                            # print("perturbated_image",perturbated_image.shape) # (1, 3, 32, 32)
                            # print("applied_patch",applied_patch.shape) # (1, 3, 32, 32)
                            # perturbated_images = perturbated_image.squeeze(0)
                            # plt.imshow(perturbated_images[0])
                            # plt.show()

                            perturbated_image = torch.from_numpy(perturbated_image)
                            # perturbated_image = torch.from_numpy(perturbated_image).cuda()
                            # print("perturbated_image",perturbated_image.shape) #torch.Size([1, 3, 32, 32])
                            perturbated_image=perturbated_image.to(self.device)

                            output= self.target_model(perturbated_image)
                            _, predicted = torch.max(output.data, 1)
                            if predicted[0].data.cpu().numpy() ==self.target:
                                train_success += 1

                            # 训练的patch
                            self.adv_patch_cpu = applied_patch[0][:, x_location:x_location + self.adv_patch_cpu.shape[1],
                                            y_location:y_location + self.adv_patch_cpu.shape[2]]
                            # print("self.adv_patch_cpu", self.adv_patch_cpu.shape)  # (3, 3, 3)
                            # plt.imshow(self.adv_patch_cpu[0])
                            # plt.show()
                # print("train_success", train_success)
            ###################################################################################

            ###################################################################################
            # print("train poison poison")
            # print("self.adv_patch_cpu_2",self.adv_patch_cpu)
            if self.data_name=='mnist' or self.data_name=='fashion_mnist':
                dataset_poison=MyDataset_mnist(self.adv_patch_cpu, active_set,self.target, self.back_prop, mode="train")
            else:
                dataset_poison = MyDataset_cifar_new(self.adv_patch_cpu, active_set,
                                                     self.target, self.back_prop, mode="train",
                                                     number_of_classes=self.number_of_classes)
            attack_data_loader_backdoor = DataLoader(dataset_poison,
                                                batch_size=self.attack_train_batch_size,
                                                shuffle=True,
                                                pin_memory=True
                                                )

            ###################################################################################

            for data, labels in attack_data_loader_backdoor:

                # Transfer data to CPU or GPU and set gradients to zero for performance.
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                self.active_attack_optimizer.zero_grad()

                # Do a forward pass through the network to get prediction values and update loss metric.
                outputs = self.target_model(data)
                loss = self.loss_function(outputs, labels)

                # Do a backward pass through the network to get the gradients
                # and then use the optimizer to update the weights.
                loss.backward()

                if is_sga>0:
                    #print("gradient ascent")
                    # torch.nn.utils.clip_grad_norm(self.target_model.parameters(), max_norm=1)
                    ####################################################
                    for name, param in self.target_model.named_parameters():
                        param.grad *= -1
                        # print("", name, param)
                    ###################################################

                self.active_attack_optimizer.step()
                losses += loss.item()
                loss_meter.update(loss.item())

        self.attack_patch_list.append(self.adv_patch_cpu)
        # Update loss and run_time metrics
        #loss_meter.update(losses)
        batch_time.update(time.time() - start_time)

        # message = f'[ Round: {round_number} ' \
        #           f'| Active Attack ' \
        #           f'| Client: {self.client_id} ' \
        #           f'| Time: {batch_time.avg:.2f}s ' \
        #           f'| Loss: {loss_meter.avg:.5f} '
        # logging.info(message)
        self.target_model.to("cpu")
        self.model = copy.deepcopy(self.target_model)


        ########################################################################
        self.model.eval()
        self.model.to(self.device)

        # load once to save time. To be used for calculating the accuracy.
        data_size = len(attack_data_loader_backdoor.dataset)

        # Define attributes to keep track of scores during testing
        batch_time = AverageMeter()
        losses = AverageMeter()
        start_time = time.time()
        correct = 0

        with torch.no_grad():
            for data, labels in attack_data_loader_backdoor:
                # Transfer data to CPU or GPU.
                data, labels = data.float().to(self.device), labels.long().to(self.device)

                # Do a forward pass through the network to get prediction values and update loss metric.
                outputs = self.model(data)
                # print(self.loss_function(outputs, labels))
                losses.update(self.loss_function(outputs, labels))

                # Compare predictions to labels and get accuracy score.
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                # Update time and accuracy metric.
                batch_time.update(time.time() - start_time)

            accuracy = (correct / data_size) * 100
            message = f'[ Round: {round_number} ' \
                      f'| Backdoor Attack Eval ' \
                      f'| Client: {self.client_id} ' \
                      f'| Loss: {losses.avg:.5f} ' \
                      f'| Backdoor Test_Acc ({correct}/{data_size})={accuracy:.2f}% ] '
            logging.info(message)
            log_attack = logging.getLogger('log_attack')
            log_attack.info(message)
        return self.adv_patch_cpu
        ########################################################################

    def get_attackmodel_loss(self, target_models):
        start_time = time.time()
        final_confusion_matrix = ConfusionMatrix()
        batch_confusion_matrix = ConfusionMatrix()
        losses = AverageMeter()
        accuracy = AverageMeter()
        # Set attack model to train
        self.attack_model.train()
        # Load in training data
        member, non_member = self.attack_train_data

        #############################
        temp_data_loader = DataLoader(member,#member
                                      batch_size=self.attack_train_batch_size // 2,
                                      shuffle=True,
                                      num_workers=0,
                                      pin_memory=True
                                      )

        temp_data_loader_2 = DataLoader(non_member,#non_member
                                        batch_size=self.attack_train_batch_size // 2,
                                        shuffle=True,
                                        num_workers=0,
                                        pin_memory=True
                                        )


        data_loader = zip(temp_data_loader, temp_data_loader_2)
        ####################################################
        # bao cun mei ge batch de ti du fan shu he loss
        l2_norm_member_list = []
        l2_norm_nonmember_list = []
        loss_member_list = []
        loss_nonmember_list = []
        temp = self.batch_size
        halftbatch = temp // 2  # xu yao geng ju batch zhi tiaozheng!!!
        times = 0
        ####################################################
        attack_loss_sum=0
        attack_loss_count=0
        for (member_input, member_target), (non_member_input, non_member_target) in data_loader:
            times += 1
            # if times==7:
            # print("round 7")
            # take timestamp
            batch_start_time = time.time()
            batch_confusion_matrix.reset()
            segmation = len(member_input)
            # Pre-define variables to use.
            model_outputs = []
            loss_values = []
            model_gradients = []
            #temp_loss_values = []
            # Load data to device
            member_input, member_target = member_input.float().to(self.device) \
                , member_target.long().to(self.device)
            non_member_input, non_member_target = non_member_input.float().to(self.device) \
                , non_member_target.long().to(self.device)

            data, labels = torch.cat((member_input, non_member_input)), torch.cat(
                (member_target, non_member_target))

            # Create one-hot encoding of labels, as this has to be done only once.
            one_hot_labels = encode_labels(labels, self.one_hot_encoding, self.device).float()


            ####################
            loss_member_eachbatch=[]
            loss_nomember_eachbatch=[]
            ####################
            # For each observed model collect input data for the attack model.
            for model in target_models:
                #
                ####################################################
                # qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                member_loss = []
                nonmember_loss = []
                ####################################################
                model.eval()
                model.to(self.device)
                optimizer = optimizers.__dict__[self.optimizer_name](
                    params=model.parameters(),
                    lr=self.learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay
                )

                # Get predictions
                predictions = model(data)
                temp_pred = torch.from_numpy(predictions.data.cpu().numpy()).to(self.device)
                model_outputs.append(temp_pred.requires_grad_(True))  # ba cheng yuan he fei chen yuan
                # calculate loss values
                # predictions 64*10 one_hot_labels 64*10
                loss_value = torch.sum(predictions * one_hot_labels, dim=1).view([-1, 1])
                ###loss_value是對應標籤的模型輸出值
                temp_loss = torch.from_numpy(loss_value.data.cpu().numpy()).to(self.device)

                loss_values.append(temp_loss.requires_grad_(True))

                gradients = torch.zeros(0)
                # dui mei yige yangebn qiu loss he tidu
                # qian 32 ge shi cheng yaun hou 32 ge fei cheng yuan
                for index in range(predictions.size(0)):
                    # ji suan yi ge shi ti de loss, qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                    loss = self.loss_function(predictions[index].view([1, -1]), labels[index].view([-1]))
                    ################## add loss into list
                    if index < segmation:
                        member_loss.append(loss)
                    else:
                        nonmember_loss.append(loss)
                    ##################
                    optimizer.zero_grad()
                    if index == (predictions.size(0)) - 1:
                        loss.backward(retain_graph=False)
                    else:
                        loss.backward(retain_graph=True)
                    # current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])#qu zui hou yi ceng can shu de tidu
                    if self.model_name == "AlexNet":
                        current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name == "CNNCifar10":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    elif self.model_name == "CNNFashion_Mnist":
                        current_grads = model.fc.weight.grad.view([1, 1, 7 * 7 * 32, self.number_of_classes])
                    elif self.model_name == "CNNMnist":
                        current_grads = model.fc2.weight.grad.view([1, 1, 50, self.number_of_classes])
                    elif self.model_name == "LeNet5":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    # current_grads = model.lastlayer_name.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name=="CNNModel":
                        current_grads = model.fc2.weight.grad.view([1, 1, 256, self.number_of_classes])
                    else:
                        current_grads = None
                    if gradients.size()[0] == 0:
                        gradients = current_grads
                    else:
                        gradients = torch.cat((gradients, current_grads))
                # 64, 1, 256, 100 qian 32 ge shi chengyuan zuihouyicengde tidu hou 32shi fei cheng yuan de tidu
                temp_gradients = torch.from_numpy(gradients.data.cpu().numpy()).to(self.device)

                model_gradients.append(temp_gradients.requires_grad_(True))
                loss_member_eachbatch.append(member_loss)
                loss_nomember_eachbatch.append(nonmember_loss)
                model.to('cpu')

            ####################################################
            ##########ji suan yi ge batch de tidu fan shu
            norm_member_eachbatch = []
            norm_nomember_eachbatch = []
            for model_lastlayer_grads in model_gradients:
                model_member_grad = model_lastlayer_grads[:segmation]  # qu 0-31

                model_nonmember_grad = model_lastlayer_grads[segmation:]  # qu 32-63
                l2_norm_member = torch.norm(model_member_grad, p=2)
                #for tmp_member in model_member_grad:
                #    print('l2norm member:',torch.norm(tmp_member, p=2))
                #for tmp_non_member in model_nonmember_grad:
                #    print('l2norm non member:',torch.norm(tmp_non_member, p=2))
                l2_norm_nonmember = torch.norm(model_nonmember_grad, p=2)
                # fang dao liang ge lie biao
                # fen bie bao cun mei yi batch de ti du fan shu
                norm_member_eachbatch.append(l2_norm_member)
                norm_nomember_eachbatch.append(l2_norm_nonmember)
            l2_norm_member_list.append(norm_member_eachbatch)
            l2_norm_nonmember_list.append(norm_nomember_eachbatch)

            loss_member_eachbatch_avg=[sum(t) / len(t) for t in loss_member_eachbatch]#get each batch avg member loss in each model
            loss_nonmember_eachbatch_avg=[sum(t) / len(t) for t in loss_nomember_eachbatch]#get each batch avg nonmember loss in each model
            loss_member_list.append(loss_member_eachbatch_avg)
            loss_nonmember_list.append(loss_nonmember_eachbatch_avg)
            ####################################################

            ####################################################
            # qian 32 ge cheng yuan hou 32 ge fei cheng yuan
            #mean_member_loss = sum(member_loss) / len(member_loss)
            # print(len(nonmember_loss))

            #mean_nonmember_loss = sum(nonmember_loss) / len(nonmember_loss)
            # print(mean_member_loss)
            # print(mean_nonmember_loss)
            #loss_member_list.append(mean_member_loss)
            #loss_nonmember_list.append(mean_nonmember_loss)
            ####################################################

            # Get the predictions for the membership classification
            self.attack_model.to(self.device)
            self.attack_optimizer.zero_grad()

            # Change labels of the data for binary attack classification
            member_target = torch.Tensor([1 for _ in member_input])
            non_member_target = torch.Tensor([0 for _ in non_member_input])
            membership_labels = torch.cat((member_target, non_member_target))
            membership_labels = torch.unsqueeze(membership_labels, -1).data.float().to(self.device)

            # Get membership predictions
            ###only model_gradients diffence
            membership_predictions = self.attack_model(model_outputs, one_hot_labels, loss_values, model_gradients)
            # Calculate the loss of attack model
            attack_loss = self.attack_loss_function(membership_predictions, membership_labels)
            attack_loss_sum+=attack_loss
            attack_loss_count+=1;
        print(attack_loss_sum/attack_loss_count)
        return attack_loss_sum/attack_loss_count;

        
            

        




    def train_attack(self, round_number, target_models):
        """ Ja hier moet dus documentatie """

        # confusion matrix and accuracy metric initialized
        start_time = time.time()
        final_confusion_matrix = ConfusionMatrix()
        batch_confusion_matrix = ConfusionMatrix()
        losses = AverageMeter()
        accuracy = AverageMeter()
        # Set attack model to train
        self.attack_model.train()
        # Load in training data
        member, non_member = self.attack_train_data
        #############################################



        #############################
        temp_data_loader = DataLoader(member,#member
                                      batch_size=self.attack_train_batch_size // 2,
                                      shuffle=True,
                                      num_workers=0,
                                      pin_memory=True
                                      )

        temp_data_loader_2 = DataLoader(non_member,#non_member
                                        batch_size=self.attack_train_batch_size // 2,
                                        shuffle=True,
                                        num_workers=0,
                                        pin_memory=True
                                        )
        

        data_loader = zip(temp_data_loader, temp_data_loader_2)
        
        ####################################################
        # bao cun mei ge batch de ti du fan shu he loss
        l2_norm_member_list = []
        l2_norm_nonmember_list = []
        loss_member_list = []
        loss_nonmember_list = []
        temp = self.batch_size
        halftbatch = temp // 2  # xu yao geng ju batch zhi tiaozheng!!!
        times = 0
        ####################################################
        for i in range(self.mia_back_rounds):
            for (member_input, member_target), (non_member_input, non_member_target) in data_loader:
                times += 1
                # if times==7:
                # print("round 7")
                # take timestamp
                batch_start_time = time.time()
                batch_confusion_matrix.reset()
                segmation = len(member_input)
                # Pre-define variables to use.
                model_outputs = []
                loss_values = []
                model_gradients = []
                #temp_loss_values = []
                # Load data to device
                member_input, member_target = member_input.float().to(self.device) \
                    , member_target.long().to(self.device)
                non_member_input, non_member_target = non_member_input.float().to(self.device) \
                    , non_member_target.long().to(self.device)

                data, labels = torch.cat((member_input, non_member_input)), torch.cat(
                    (member_target, non_member_target))

                # Create one-hot encoding of labels, as this has to be done only once.
                one_hot_labels = encode_labels(labels, self.one_hot_encoding, self.device).float()


                ####################
                loss_member_eachbatch=[]
                loss_nomember_eachbatch=[]
                ####################
                # For each observed model collect input data for the attack model.
                for model in target_models:
                    #
                    ####################################################
                    # qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                    member_loss = []
                    nonmember_loss = []
                    ####################################################
                    model.eval()
                    model.to(self.device)
                    optimizer = optimizers.__dict__[self.optimizer_name](
                        params=model.parameters(),
                        lr=self.learning_rate,
                        momentum=self.momentum,
                        weight_decay=self.weight_decay
                    )

                    # Get predictions
                    predictions = model(data)
                    temp_pred = torch.from_numpy(predictions.data.cpu().numpy()).to(self.device)
                    model_outputs.append(temp_pred.requires_grad_(True))  # ba cheng yuan he fei chen yuan
                    # calculate loss values
                    # predictions 64*10 one_hot_labels 64*10
                    loss_value = torch.sum(predictions * one_hot_labels, dim=1).view([-1, 1])
                    ###loss_value是對應標籤的模型輸出值
                    temp_loss = torch.from_numpy(loss_value.data.cpu().numpy()).to(self.device)

                    loss_values.append(temp_loss.requires_grad_(True))

                    gradients = torch.zeros(0)
                    # dui mei yige yangebn qiu loss he tidu
                    # qian 32 ge shi cheng yaun hou 32 ge fei cheng yuan
                    for index in range(predictions.size(0)):
                        # ji suan yi ge shi ti de loss, qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                        loss = self.loss_function(predictions[index].view([1, -1]), labels[index].view([-1]))
                        ################## add loss into list
                        if index < segmation:
                            member_loss.append(loss)
                        else:
                            nonmember_loss.append(loss)
                        ##################
                        optimizer.zero_grad()
                        if index == (predictions.size(0)) - 1:
                            loss.backward(retain_graph=False)
                        else:
                            loss.backward(retain_graph=True)
                        # current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])#qu zui hou yi ceng can shu de tidu
                        if self.model_name == "AlexNet":
                            current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])
                        elif self.model_name == "CNNCifar10":
                            current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                        elif self.model_name == "CNNFashion_Mnist":
                            current_grads = model.fc.weight.grad.view([1, 1, 7 * 7 * 32, self.number_of_classes])
                        elif self.model_name == "CNNMnist":
                            current_grads = model.fc2.weight.grad.view([1, 1, 50, self.number_of_classes])
                        elif self.model_name == "LeNet5":
                            current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                        # current_grads = model.lastlayer_name.weight.grad.view([1, 1, 256, self.number_of_classes])
                        elif self.model_name=="CNNModel":
                            current_grads = model.fc2.weight.grad.view([1, 1, 256, self.number_of_classes])
                        else:
                            current_grads = None
                        if gradients.size()[0] == 0:
                            gradients = current_grads
                        else:
                            gradients = torch.cat((gradients, current_grads))
                    # 64, 1, 256, 100 qian 32 ge shi chengyuan zuihouyicengde tidu hou 32shi fei cheng yuan de tidu
                    temp_gradients = torch.from_numpy(gradients.data.cpu().numpy()).to(self.device)

                    model_gradients.append(temp_gradients.requires_grad_(True))
                    loss_member_eachbatch.append(member_loss)
                    loss_nomember_eachbatch.append(nonmember_loss)
                    model.to('cpu')

                ####################################################
                ##########ji suan yi ge batch de tidu fan shu
                norm_member_eachbatch = []
                norm_nomember_eachbatch = []
                for model_lastlayer_grads in model_gradients:
                    model_member_grad = model_lastlayer_grads[:segmation]  # qu 0-31

                    model_nonmember_grad = model_lastlayer_grads[segmation:]  # qu 32-63
                    l2_norm_member = torch.norm(model_member_grad, p=2)
                    #for tmp_member in model_member_grad:
                    #    print('l2norm member:',torch.norm(tmp_member, p=2))
                    #for tmp_non_member in model_nonmember_grad:
                    #    print('l2norm non member:',torch.norm(tmp_non_member, p=2))
                    l2_norm_nonmember = torch.norm(model_nonmember_grad, p=2)
                    # fang dao liang ge lie biao
                    # fen bie bao cun mei yi batch de ti du fan shu
                    norm_member_eachbatch.append(l2_norm_member)
                    norm_nomember_eachbatch.append(l2_norm_nonmember)
                l2_norm_member_list.append(norm_member_eachbatch)
                l2_norm_nonmember_list.append(norm_nomember_eachbatch)

                loss_member_eachbatch_avg=[sum(t) / len(t) for t in loss_member_eachbatch]#get each batch avg member loss in each model
                loss_nonmember_eachbatch_avg=[sum(t) / len(t) for t in loss_nomember_eachbatch]#get each batch avg nonmember loss in each model
                loss_member_list.append(loss_member_eachbatch_avg)
                loss_nonmember_list.append(loss_nonmember_eachbatch_avg)
                ####################################################

                ####################################################
                # qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                #mean_member_loss = sum(member_loss) / len(member_loss)
                # print(len(nonmember_loss))

                #mean_nonmember_loss = sum(nonmember_loss) / len(nonmember_loss)
                # print(mean_member_loss)
                # print(mean_nonmember_loss)
                #loss_member_list.append(mean_member_loss)
                #loss_nonmember_list.append(mean_nonmember_loss)
                ####################################################

                # Get the predictions for the membership classification
                self.attack_model.to(self.device)
                self.attack_optimizer.zero_grad()

                # Change labels of the data for binary attack classification
                member_target = torch.Tensor([1 for _ in member_input])
                non_member_target = torch.Tensor([0 for _ in non_member_input])
                membership_labels = torch.cat((member_target, non_member_target))
                membership_labels = torch.unsqueeze(membership_labels, -1).data.float().to(self.device)

                # Get membership predictions
                ###only model_gradients diffence
                membership_predictions = self.attack_model(model_outputs, one_hot_labels, loss_values, model_gradients)
                # Calculate the loss of attack model
                attack_loss = self.attack_loss_function(membership_predictions, membership_labels)

                # Perform optimizer steps
                attack_loss.backward()
                ###after backward() habe grad
                self.attack_optimizer.step()

                # Place model back to CPU
                self.attack_model.to('cpu')

                ####################################################
                '''
                #求最后一层参数的梯度范数
                params = list(model.named_parameters())
                last_layer_parameters = params[-1]

                last_layer_grad = last_layer_parameters[1].grad
                l2_norm = torch.norm(last_layer_grad, p=2)  # 计算L2范数
                '''
                ####################################################

                # Measure training accuracy and report metric
                batch_confusion_matrix.update(membership_predictions, membership_labels)
                accuracy.update(batch_confusion_matrix.get_accuracy(), self.attack_train_batch_size)
                losses.update(attack_loss.item(), self.attack_train_batch_size)

                temp = batch_confusion_matrix.get_confusion_matrix()
                final_confusion_matrix.update_from_matrix(temp)

                message = f'[ Round: {round_number} ' \
                        f'| Attacker Train ' \
                        f'| Batch Time: {time.time() - batch_start_time:.2f}s ' \
                        f'| Batch Loss: {attack_loss:.5f} ' \
                        f'| Batch Acc: {batch_confusion_matrix.get_accuracy():.2f}% ]' \
                        f'| Batch Conf Matrix: TP:{temp[0]}' \
                        f' FP:{temp[1]}' \
                        f' TN:{temp[2]}' \
                        f' FN:{temp[3]} ]'


                logging.info(message)
                if times%self.save_information_count==0:
                    log_attack = logging.getLogger('log_attack')
                    log_attack.info(message)
        
        
    

        

        l2_norm_member_round_avg=[sum(x) / len(x) for x in zip(*l2_norm_member_list)]
        l2_norm_nonmember_round_avg=[sum(x) / len(x) for x in zip(*l2_norm_nonmember_list)]
        loss_member_round_avg=[sum(x) / len(x) for x in zip(*loss_member_list)]
        loss_nonmember_round_avg=[sum(x) / len(x) for x in zip(*loss_nonmember_list)]
        decimal_places=4
        l2_norm_member_round_avg = [round(tensor.item(), decimal_places) for tensor in l2_norm_member_round_avg]
        l2_norm_nonmember_round_avg = [round(tensor.item(), decimal_places) for tensor in l2_norm_nonmember_round_avg]
        loss_member_round_avg = [round(tensor.item(), decimal_places) for tensor in loss_member_round_avg]
        loss_nonmember_round_avg = [round(tensor.item(), decimal_places) for tensor in loss_nonmember_round_avg]

       

        ####################################################

        temp = final_confusion_matrix.get_confusion_matrix()
        self.best_inference_acc_train = max(final_confusion_matrix.get_accuracy(),self.best_inference_acc_train)
        message = f'[ Round: {round_number} Totals ' \
                  f'| Attacker Train ' \
                  f'| Total Time: {time.time() - start_time:.2f}s ' \
                  f'| Avg Loss: {losses.avg:.5f} ' \
                  f'| Grad Norm Member: {l2_norm_member_round_avg} ' \
                  f'| Grad Norm NonMember: {l2_norm_nonmember_round_avg} ' \
                  f'| Loss Member: {loss_member_round_avg} ' \
                  f'| Loss NonMember: {loss_nonmember_round_avg} ' \
                  f'| Acc: {final_confusion_matrix.get_accuracy():.2f}% ]' \
                  f'| Final Conf Matrix: TP:{temp[0]}' \
                  f' FP:{temp[1]}' \
                  f' TN:{temp[2]}' \
                  f' FN:{temp[3]} ]'


        logging.info(message)
        log_attack = logging.getLogger('log_attack')
        log_attack.info(message)



    
    def train_attack1(self, round_number, target_models):
        """ Ja hier moet dus documentatie """

        # confusion matrix and accuracy metric initialized
        start_time = time.time()
        final_confusion_matrix = ConfusionMatrix()
        batch_confusion_matrix = ConfusionMatrix()
        losses = AverageMeter()
        accuracy = AverageMeter()
        # Set attack model to train
        self.attack_model.train()
        # Load in training data
        member, non_member = self.attack_train_data
        #############################################
        member_poison=self.attack_train_data_member_poison
        non_member_poison=self.attack_train_non_member_poison
        #############################
        '''
        if self.data_name == 'mnist' or self.data_name == 'fashion_mnist':
            dataset_poison_member = MyDataset_mnist(self.adv_patch_cpu, member, self.target, 1, mode="test")
            dataset_poison_nonmember = MyDataset_mnist(self.adv_patch_cpu, non_member, self.target, 1, mode="test")
        else:
            dataset_poison_member = MyDataset_cifar_new(self.adv_patch_cpu, member, self.target, 1, mode="test")
            dataset_poison_nonmember = MyDataset_cifar_new(self.adv_patch_cpu, non_member, self.target, 1, mode="test")
        '''


        #############################
        temp_data_loader = DataLoader(member,#member
                                      batch_size=self.attack_train_batch_size // 2,
                                      shuffle=False,
                                      num_workers=0,
                                      pin_memory=True
                                      )

        temp_data_loader_2 = DataLoader(non_member,#non_member
                                        batch_size=self.attack_train_batch_size // 2,
                                        shuffle=False,
                                        num_workers=0,
                                        pin_memory=True
                                        )

        temp_data_loader_poison=DataLoader(member_poison,
                                        batch_size=self.attack_train_batch_size // 2,
                                        shuffle=False,
                                        num_workers=0,
                                        pin_memory=True
        )


        temp_data_loader_2_poison=DataLoader(non_member_poison,
                                        batch_size=self.attack_train_batch_size // 2,
                                        shuffle=False,
                                        num_workers=0,
                                        pin_memory=True
        )


        data_loader = zip(temp_data_loader, temp_data_loader_2,temp_data_loader_poison,temp_data_loader_2_poison)
        #data_loader_poison=zip(temp_data_loader_poison,temp_data_loader_2_poison)
        ####################################################
        # bao cun mei ge batch de ti du fan shu he loss
        l2_norm_member_list = []
        l2_norm_nonmember_list = []
        loss_member_list = []
        loss_nonmember_list = []



        l2_norm_member_poison_list = []
        l2_norm_nonmember_poison_list = []
        loss_member_poison_list = []
        loss_nonmember_poison_list = []


        temp = self.batch_size
        halftbatch = temp // 2  # xu yao geng ju batch zhi tiaozheng!!!
        times = 0
        ####################################################
        for (member_input, member_target), (non_member_input, non_member_target),(member_input_poison, member_target_poison),(non_member_input_poison, non_member_target_poison) in data_loader:
            times += 1
            # if times==7:
            # print("round 7")
            # take timestamp
            batch_start_time = time.time()
            batch_confusion_matrix.reset()
            segmation = len(member_input)
            # Pre-define variables to use.
            model_outputs = []
            loss_values = []
            model_gradients = []



            model_outputs_poison = []
            loss_values_poison = []
            model_gradients_poison = []

            #temp_loss_values = []
            # Load data to device
            member_input, member_target = member_input.float().to(self.device) \
                , member_target.long().to(self.device)
            non_member_input, non_member_target = non_member_input.float().to(self.device) \
                , non_member_target.long().to(self.device)
            
            member_input_poison, member_target_poison = member_input_poison.float().to(self.device) \
                , member_target_poison.long().to(self.device)
            non_member_input_poison, non_member_target_poison = non_member_input_poison.float().to(self.device) \
                , non_member_target_poison.long().to(self.device)

            data, labels = torch.cat((member_input, non_member_input)), torch.cat(
                (member_target, non_member_target))

            # Create one-hot encoding of labels, as this has to be done only once.
            one_hot_labels = encode_labels(labels, self.one_hot_encoding, self.device).float()

            data_poison, labels_poison = torch.cat((member_input_poison, non_member_input_poison)), torch.cat(
                (member_target_poison, non_member_target_poison))
            #plt.imshow(data_poison[0].cpu().numpy().transpose(1,2,0))
            #plt.show()
            one_hot_labels_poison = encode_labels(labels_poison, self.one_hot_encoding, self.device).float()
            ####################
            loss_member_eachbatch=[]
            loss_nomember_eachbatch=[]

            loss_member_eachbatch_poison=[]
            loss_nomember_eachbatch_poison=[]
            ####################
            # For each observed model collect input data for the attack model.

            loss_labal_ori_list=[]

            for model in target_models:
                #
                ####################################################
                # qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                member_loss = []
                nonmember_loss = []
                ####################################################
                model.eval()
                model.to(self.device)
                optimizer = optimizers.__dict__[self.optimizer_name](
                    params=model.parameters(),
                    lr=self.learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay
                )

                # Get predictions
                predictions = model(data)
                temp_pred = torch.from_numpy(predictions.data.cpu().numpy()).to(self.device)
                model_outputs.append(temp_pred.requires_grad_(True))  # ba cheng yuan he fei chen yuan
                # calculate loss values
                # predictions 64*10 one_hot_labels 64*10
                loss_value = torch.sum(predictions * one_hot_labels, dim=1).view([-1, 1])
                ###loss_value是對應標籤的模型輸出值
                temp_loss = torch.from_numpy(loss_value.data.cpu().numpy()).to(self.device)

                loss_values.append(temp_loss.requires_grad_(True))
                loss_labal=[]
                gradients = torch.zeros(0)
                # dui mei yige yangebn qiu loss he tidu
                # qian 32 ge shi cheng yaun hou 32 ge fei cheng yuan
                for index in range(predictions.size(0)):
                    # ji suan yi ge shi ti de loss, qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                    loss = self.loss_function(predictions[index].view([1, -1]), labels[index].view([-1]))
                    loss_labal.append(loss)
                    ################## add loss into list
                    if index < segmation:
                        member_loss.append(loss)
                    else:
                        nonmember_loss.append(loss)
                    ##################
                    optimizer.zero_grad()
                    if index == (predictions.size(0)) - 1:
                        loss.backward(retain_graph=False)
                    else:
                        loss.backward(retain_graph=True)
                    # current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])#qu zui hou yi ceng can shu de tidu
                    if self.model_name == "AlexNet":
                        current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name == "CNNCifar10":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    elif self.model_name == "CNNFashion_Mnist":
                        current_grads = model.fc.weight.grad.view([1, 1, 7 * 7 * 32, self.number_of_classes])
                    elif self.model_name == "CNNMnist":
                        current_grads = model.fc2.weight.grad.view([1, 1, 50, self.number_of_classes])
                    elif self.model_name == "LeNet5":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    # current_grads = model.lastlayer_name.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name=="CNNModel":
                        current_grads = model.fc2.weight.grad.view([1, 1, 256, self.number_of_classes])
                    else:
                        current_grads = None
                    if gradients.size()[0] == 0:
                        gradients = current_grads
                    else:
                        gradients = torch.cat((gradients, current_grads))
                # 64, 1, 256, 100 qian 32 ge shi chengyuan zuihouyicengde tidu hou 32shi fei cheng yuan de tidu
                temp_gradients = torch.from_numpy(gradients.data.cpu().numpy()).to(self.device)
                loss_labal_ori_list.append(torch.unsqueeze(torch.tensor(loss_labal), dim=1))

                model_gradients.append(temp_gradients.requires_grad_(True))
                loss_member_eachbatch.append(member_loss)
                loss_nomember_eachbatch.append(nonmember_loss)
                model.to('cpu')
            



            loss_labal_poison_list=[]
            for model in target_models:
                #
                ####################################################
                # qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                member_loss_poison = []
                nonmember_loss_poison = []
                ####################################################
                model.eval()
                model.to(self.device)
                optimizer = optimizers.__dict__[self.optimizer_name](
                    params=model.parameters(),
                    lr=self.learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay
                )

                # Get predictions
                predictions = model(data_poison)
                _, predicted = torch.max(predictions, 1)

                temp_pred = torch.from_numpy(predictions.data.cpu().numpy()).to(self.device)
                model_outputs_poison.append(temp_pred.requires_grad_(True))  # ba cheng yuan he fei chen yuan
                # calculate loss values
                # predictions 64*10 one_hot_labels 64*10
                loss_value = torch.sum(predictions * one_hot_labels_poison, dim=1).view([-1, 1])

                ###loss_value是對應標籤的模型輸出值
                temp_loss = torch.from_numpy(loss_value.data.cpu().numpy()).to(self.device)

                loss_values_poison.append(temp_loss.requires_grad_(True))

                gradients = torch.zeros(0)
                # dui mei yige yangebn qiu loss he tidu
                # qian 32 ge shi cheng yaun hou 32 ge fei cheng yuan
                loss_labal=[]
                for index in range(predictions.size(0)):
                    # ji suan yi ge shi ti de loss, qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                    loss = self.loss_function(predictions[index].view([1, -1]), labels[index].view([-1]))
                    loss_labal.append(loss)
                    ################## add loss into list
                    if index < segmation:
                        member_loss_poison.append(loss)
                    else:
                        nonmember_loss_poison.append(loss)
                    ##################
                    optimizer.zero_grad()
                    if index == (predictions.size(0)) - 1:
                        loss.backward(retain_graph=False)
                    else:
                        loss.backward(retain_graph=True)
                    # current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])#qu zui hou yi ceng can shu de tidu
                    if self.model_name == "AlexNet":
                        current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name == "CNNCifar10":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    elif self.model_name == "CNNFashion_Mnist":
                        current_grads = model.fc.weight.grad.view([1, 1, 7 * 7 * 32, self.number_of_classes])
                    elif self.model_name == "CNNMnist":
                        current_grads = model.fc2.weight.grad.view([1, 1, 50, self.number_of_classes])
                    elif self.model_name == "LeNet5":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    # current_grads = model.lastlayer_name.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name=="CNNModel":
                        current_grads = model.fc2.weight.grad.view([1, 1, 256, self.number_of_classes])
                    else:
                        current_grads = None
                    if gradients.size()[0] == 0:
                        gradients = current_grads
                    else:
                        gradients = torch.cat((gradients, current_grads))
                # 64, 1, 256, 100 qian 32 ge shi chengyuan zuihouyicengde tidu hou 32shi fei cheng yuan de tidu
                temp_gradients = torch.from_numpy(gradients.data.cpu().numpy()).to(self.device)
                loss_labal_poison_list.append(torch.unsqueeze(torch.tensor(loss_labal), dim=1))
                model_gradients_poison.append(temp_gradients.requires_grad_(True))
                loss_member_eachbatch_poison.append(member_loss)
                loss_nomember_eachbatch_poison.append(nonmember_loss)
                model.to('cpu')



            ####################################################
            ##########ji suan yi ge batch de tidu fan shu
            norm_member_eachbatch = []
            norm_nomember_eachbatch = []

            norm_member_eachbatch_poison = []
            norm_nomember_eachbatch_poison = []

            for model_lastlayer_grads in model_gradients:
                model_member_grad = model_lastlayer_grads[:segmation]  # qu 0-31

                model_nonmember_grad = model_lastlayer_grads[segmation:]  # qu 32-63
                l2_norm_member = torch.norm(model_member_grad, p=2)
                #for tmp_member in model_member_grad:
                #    print('l2norm member:',torch.norm(tmp_member, p=2))
                #for tmp_non_member in model_nonmember_grad:
                #    print('l2norm non member:',torch.norm(tmp_non_member, p=2))
                l2_norm_nonmember = torch.norm(model_nonmember_grad, p=2)
                # fang dao liang ge lie biao
                # fen bie bao cun mei yi batch de ti du fan shu
                norm_member_eachbatch.append(l2_norm_member)
                norm_nomember_eachbatch.append(l2_norm_nonmember)
            l2_norm_member_list.append(norm_member_eachbatch)
            l2_norm_nonmember_list.append(norm_nomember_eachbatch)


            for model_lastlayer_grads in model_gradients_poison:
                model_member_grad = model_lastlayer_grads[:segmation]  # qu 0-31

                model_nonmember_grad = model_lastlayer_grads[segmation:]  # qu 32-63
                l2_norm_member = torch.norm(model_member_grad, p=2)
                #for tmp_member in model_member_grad:
                #    print('l2norm member:',torch.norm(tmp_member, p=2))
                #for tmp_non_member in model_nonmember_grad:
                #    print('l2norm non member:',torch.norm(tmp_non_member, p=2))
                l2_norm_nonmember = torch.norm(model_nonmember_grad, p=2)
                # fang dao liang ge lie biao
                # fen bie bao cun mei yi batch de ti du fan shu
                norm_member_eachbatch_poison.append(l2_norm_member)
                norm_nomember_eachbatch_poison.append(l2_norm_nonmember)
            l2_norm_member_poison_list.append(loss_member_eachbatch_poison)
            l2_norm_nonmember_poison_list.append(norm_nomember_eachbatch_poison)            

            loss_member_eachbatch_avg=[sum(t) / len(t) for t in loss_member_eachbatch]#get each batch avg member loss in each model
            loss_nonmember_eachbatch_avg=[sum(t) / len(t) for t in loss_nomember_eachbatch]#get each batch avg nonmember loss in each model
            loss_member_list.append(loss_member_eachbatch_avg)
            loss_nonmember_list.append(loss_nonmember_eachbatch_avg)


            loss_member_eachbatch_avg_poison=[sum(t) / len(t) for t in loss_member_eachbatch_poison]#get each batch avg member loss in each model
            loss_nonmember_eachbatch_avg_poison=[sum(t) / len(t) for t in loss_nomember_eachbatch_poison]#get each batch avg nonmember loss in each model
            loss_member_poison_list.append(loss_member_eachbatch_avg_poison)
            loss_nonmember_poison_list.append(loss_nonmember_eachbatch_avg_poison)

            diff_loss=[a-b for a,b in zip(loss_labal_ori_list,loss_labal_poison_list)]
            diff_grad=[a-b for a,b in zip(model_gradients,model_gradients_poison)]

            for i in range(len(diff_loss)):
                diff_loss[i]=diff_loss[i].to(self.device)
            for i in range(len(diff_grad)):
                diff_grad[i]=diff_grad[i].to(self.device)


            

            #print()
            #print()
            
            ####################################################

            ####################################################
            # qian 32 ge cheng yuan hou 32 ge fei cheng yuan
            #mean_member_loss = sum(member_loss) / len(member_loss)
            # print(len(nonmember_loss))

            #mean_nonmember_loss = sum(nonmember_loss) / len(nonmember_loss)
            # print(mean_member_loss)
            # print(mean_nonmember_loss)
            #loss_member_list.append(mean_member_loss)
            #loss_nonmember_list.append(mean_nonmember_loss)
            ####################################################

            # Get the predictions for the membership classification
            self.attack_model.to(self.device)
            self.attack_optimizer.zero_grad()

            # Change labels of the data for binary attack classification
            member_target = torch.Tensor([1 for _ in member_input])
            non_member_target = torch.Tensor([0 for _ in non_member_input])
            membership_labels = torch.cat((member_target, non_member_target))
            membership_labels = torch.unsqueeze(membership_labels, -1).data.float().to(self.device)

            # Get membership predictions
            ###only model_gradients diffence
            membership_predictions = self.attack_model(model_outputs, one_hot_labels, loss_values, model_gradients,diff_loss,diff_grad)
            # Calculate the loss of attack model
            attack_loss = self.attack_loss_function(membership_predictions, membership_labels)

            # Perform optimizer steps
            attack_loss.backward()
            ###after backward() habe grad
            self.attack_optimizer.step()

            # Place model back to CPU
            self.attack_model.to('cpu')

            # Measure training accuracy and report metric
            batch_confusion_matrix.update(membership_predictions, membership_labels)
            accuracy.update(batch_confusion_matrix.get_accuracy(), self.attack_train_batch_size)
            losses.update(attack_loss.item(), self.attack_train_batch_size)

            temp = batch_confusion_matrix.get_confusion_matrix()
            final_confusion_matrix.update_from_matrix(temp)

            message = f'[ Round: {round_number} ' \
                      f'| Attacker Train ' \
                      f'| Batch Time: {time.time() - batch_start_time:.2f}s ' \
                      f'| Batch Loss: {attack_loss:.5f} ' \
                      f'| Batch Acc: {batch_confusion_matrix.get_accuracy():.2f}% ]' \
                      f'| Batch Conf Matrix: TP:{temp[0]}' \
                      f' FP:{temp[1]}' \
                      f' TN:{temp[2]}' \
                      f' FN:{temp[3]} ]'


            logging.info(message)
            if times%self.save_information_count==0:
                log_attack = logging.getLogger('log_attack')
                log_attack.info(message)
        
        '''
        times = 0
        temp_data_loader_trigger = DataLoader(member,#member
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=0,
                                      pin_memory=True
                                      )

        temp_data_loader_2_trigger = DataLoader(non_member,#non_member
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=0,
                                        pin_memory=True
                                        )

        temp_data_loader_poison_trigger=DataLoader(member_poison,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=0,
                                        pin_memory=True
        )


        temp_data_loader_2_poison_trigger=DataLoader(non_member_poison,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=0,
                                        pin_memory=True
        )

        data_loader_trigger = zip(temp_data_loader_trigger, temp_data_loader_2_trigger,temp_data_loader_poison_trigger,temp_data_loader_2_poison_trigger)

        for (member_input, member_target), (non_member_input, non_member_target),(member_input_poison, member_target_poison),(non_member_input_poison, non_member_target_poison) in data_loader_trigger:
            times += 1
            # if times==7:
            # print("round 7")
            # take timestamp
            batch_start_time = time.time()
            batch_confusion_matrix.reset()
            segmation = len(member_input)
            # Pre-define variables to use.
            model_outputs = []
            loss_values = []
            model_gradients = []



            model_outputs_poison = []
            loss_values_poison = []
            model_gradients_poison = []

            #temp_loss_values = []
            # Load data to device
            member_input, member_target = member_input.float().to(self.device) \
                , member_target.long().to(self.device)
            non_member_input, non_member_target = non_member_input.float().to(self.device) \
                , non_member_target.long().to(self.device)
            
            member_input_poison, member_target_poison = member_input_poison.float().to(self.device) \
                , member_target_poison.long().to(self.device)
            non_member_input_poison, non_member_target_poison = non_member_input_poison.float().to(self.device) \
                , non_member_target_poison.long().to(self.device)
            


            data, labels = torch.cat((member_input, non_member_input)), torch.cat(
                (member_target, non_member_target))

            # Create one-hot encoding of labels, as this has to be done only once.
            one_hot_labels = encode_labels(labels, self.one_hot_encoding, self.device).float()

            data_poison, labels_poison = torch.cat((member_input_poison, non_member_input_poison)), torch.cat(
                (member_target_poison, non_member_target_poison))
            #plt.imshow(data_poison[0].cpu().numpy().transpose(1,2,0))
            #plt.show()
            one_hot_labels_poison = encode_labels(labels_poison, self.one_hot_encoding, self.device).float()
            ####################
            loss_member_eachbatch=[]
            loss_nomember_eachbatch=[]

            loss_member_eachbatch_poison=[]
            loss_nomember_eachbatch_poison=[]
            ####################
            # For each observed model collect input data for the attack model.

            loss_labal_ori_list=[]

            for model in target_models:
                #
                ####################################################
                # qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                member_loss = []
                nonmember_loss = []
                ####################################################
                model.eval()
                model.to(self.device)
                optimizer = optimizers.__dict__[self.optimizer_name](
                    params=model.parameters(),
                    lr=self.learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay
                )

                # Get predictions
                predictions = model(data)
                temp_pred = torch.from_numpy(predictions.data.cpu().numpy()).to(self.device)
                model_outputs.append(temp_pred.requires_grad_(True))  # ba cheng yuan he fei chen yuan
                # calculate loss values
                # predictions 64*10 one_hot_labels 64*10
                loss_value = torch.sum(predictions * one_hot_labels, dim=1).view([-1, 1])
                ###loss_value是對應標籤的模型輸出值
                temp_loss = torch.from_numpy(loss_value.data.cpu().numpy()).to(self.device)

                loss_values.append(temp_loss.requires_grad_(True))
                loss_labal=[]
                gradients = torch.zeros(0)
                # dui mei yige yangebn qiu loss he tidu
                # qian 32 ge shi cheng yaun hou 32 ge fei cheng yuan
                for index in range(predictions.size(0)):
                    # ji suan yi ge shi ti de loss, qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                    loss = self.loss_function(predictions[index].view([1, -1]), labels[index].view([-1]))
                    loss_labal.append(loss)
                    ################## add loss into list
                    if index < segmation:
                        member_loss.append(loss)
                    else:
                        nonmember_loss.append(loss)
                    ##################
                    optimizer.zero_grad()
                    if index == (predictions.size(0)) - 1:
                        loss.backward(retain_graph=False)
                    else:
                        loss.backward(retain_graph=True)
                    # current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])#qu zui hou yi ceng can shu de tidu
                    if self.model_name == "AlexNet":
                        current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name == "CNNCifar10":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    elif self.model_name == "CNNFashion_Mnist":
                        current_grads = model.fc.weight.grad.view([1, 1, 7 * 7 * 32, self.number_of_classes])
                    elif self.model_name == "CNNMnist":
                        current_grads = model.fc2.weight.grad.view([1, 1, 50, self.number_of_classes])
                    elif self.model_name == "LeNet5":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    # current_grads = model.lastlayer_name.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name=="CNNModel":
                        current_grads = model.fc2.weight.grad.view([1, 1, 256, self.number_of_classes])
                    else:
                        current_grads = None
                    if gradients.size()[0] == 0:
                        gradients = current_grads
                    else:
                        gradients = torch.cat((gradients, current_grads))
                # 64, 1, 256, 100 qian 32 ge shi chengyuan zuihouyicengde tidu hou 32shi fei cheng yuan de tidu
                temp_gradients = torch.from_numpy(gradients.data.cpu().numpy()).to(self.device)
                loss_labal_ori_list.append(torch.unsqueeze(torch.tensor(loss_labal), dim=1))

                model_gradients.append(temp_gradients.requires_grad_(True))
                loss_member_eachbatch.append(member_loss)
                loss_nomember_eachbatch.append(nonmember_loss)
                model.to('cpu')
            



            loss_labal_poison_list=[]
            for model in target_models:
                #
                ####################################################
                # qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                member_loss_poison = []
                nonmember_loss_poison = []
                ####################################################
                model.eval()
                model.to(self.device)
                optimizer = optimizers.__dict__[self.optimizer_name](
                    params=model.parameters(),
                    lr=self.learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay
                )

                # Get predictions
                predictions = model(data_poison)
                _, predicted = torch.max(predictions, 1)

                temp_pred = torch.from_numpy(predictions.data.cpu().numpy()).to(self.device)
                model_outputs_poison.append(temp_pred.requires_grad_(True))  # ba cheng yuan he fei chen yuan
                # calculate loss values
                # predictions 64*10 one_hot_labels 64*10
                loss_value = torch.sum(predictions * one_hot_labels_poison, dim=1).view([-1, 1])

                ###loss_value是對應標籤的模型輸出值
                temp_loss = torch.from_numpy(loss_value.data.cpu().numpy()).to(self.device)

                loss_values_poison.append(temp_loss.requires_grad_(True))

                gradients = torch.zeros(0)
                # dui mei yige yangebn qiu loss he tidu
                # qian 32 ge shi cheng yaun hou 32 ge fei cheng yuan
                loss_labal=[]
                for index in range(predictions.size(0)):
                    # ji suan yi ge shi ti de loss, qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                    loss = self.loss_function(predictions[index].view([1, -1]), labels[index].view([-1]))
                    loss_labal.append(loss)
                    ################## add loss into list
                    if index < segmation:
                        member_loss_poison.append(loss)
                    else:
                        nonmember_loss_poison.append(loss)
                    ##################
                    optimizer.zero_grad()
                    if index == (predictions.size(0)) - 1:
                        loss.backward(retain_graph=False)
                    else:
                        loss.backward(retain_graph=True)
                    # current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])#qu zui hou yi ceng can shu de tidu
                    if self.model_name == "AlexNet":
                        current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name == "CNNCifar10":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    elif self.model_name == "CNNFashion_Mnist":
                        current_grads = model.fc.weight.grad.view([1, 1, 7 * 7 * 32, self.number_of_classes])
                    elif self.model_name == "CNNMnist":
                        current_grads = model.fc2.weight.grad.view([1, 1, 50, self.number_of_classes])
                    elif self.model_name == "LeNet5":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    # current_grads = model.lastlayer_name.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name=="CNNModel":
                        current_grads = model.fc2.weight.grad.view([1, 1, 256, self.number_of_classes])
                    else:
                        current_grads = None
                    if gradients.size()[0] == 0:
                        gradients = current_grads
                    else:
                        gradients = torch.cat((gradients, current_grads))
                # 64, 1, 256, 100 qian 32 ge shi chengyuan zuihouyicengde tidu hou 32shi fei cheng yuan de tidu
                temp_gradients = torch.from_numpy(gradients.data.cpu().numpy()).to(self.device)
                loss_labal_poison_list.append(torch.unsqueeze(torch.tensor(loss_labal), dim=1))
                model_gradients_poison.append(temp_gradients.requires_grad_(True))
                loss_member_eachbatch_poison.append(member_loss)
                loss_nomember_eachbatch_poison.append(nonmember_loss)
                model.to('cpu')



            ####################################################
            ##########ji suan yi ge batch de tidu fan shu
            norm_member_eachbatch = []
            norm_nomember_eachbatch = []

            norm_member_eachbatch_poison = []
            norm_nomember_eachbatch_poison = []

            for model_lastlayer_grads in model_gradients:
                model_member_grad = model_lastlayer_grads[:segmation]  # qu 0-31

                model_nonmember_grad = model_lastlayer_grads[segmation:]  # qu 32-63
                l2_norm_member = torch.norm(model_member_grad, p=2)
                #for tmp_member in model_member_grad:
                #    print('l2norm member:',torch.norm(tmp_member, p=2))
                #for tmp_non_member in model_nonmember_grad:
                #    print('l2norm non member:',torch.norm(tmp_non_member, p=2))
                l2_norm_nonmember = torch.norm(model_nonmember_grad, p=2)
                # fang dao liang ge lie biao
                # fen bie bao cun mei yi batch de ti du fan shu
                norm_member_eachbatch.append(l2_norm_member)
                norm_nomember_eachbatch.append(l2_norm_nonmember)
            l2_norm_member_list.append(norm_member_eachbatch)
            l2_norm_nonmember_list.append(norm_nomember_eachbatch)


            for model_lastlayer_grads in model_gradients_poison:
                model_member_grad = model_lastlayer_grads[:segmation]  # qu 0-31

                model_nonmember_grad = model_lastlayer_grads[segmation:]  # qu 32-63
                l2_norm_member = torch.norm(model_member_grad, p=2)
                #for tmp_member in model_member_grad:
                #    print('l2norm member:',torch.norm(tmp_member, p=2))
                #for tmp_non_member in model_nonmember_grad:
                #    print('l2norm non member:',torch.norm(tmp_non_member, p=2))
                l2_norm_nonmember = torch.norm(model_nonmember_grad, p=2)
                # fang dao liang ge lie biao
                # fen bie bao cun mei yi batch de ti du fan shu
                norm_member_eachbatch_poison.append(l2_norm_member)
                norm_nomember_eachbatch_poison.append(l2_norm_nonmember)
            l2_norm_member_poison_list.append(loss_member_eachbatch_poison)
            l2_norm_nonmember_poison_list.append(norm_nomember_eachbatch_poison)            

            loss_member_eachbatch_avg=[sum(t) / len(t) for t in loss_member_eachbatch]#get each batch avg member loss in each model
            loss_nonmember_eachbatch_avg=[sum(t) / len(t) for t in loss_nomember_eachbatch]#get each batch avg nonmember loss in each model
            loss_member_list.append(loss_member_eachbatch_avg)
            loss_nonmember_list.append(loss_nonmember_eachbatch_avg)


            loss_member_eachbatch_avg_poison=[sum(t) / len(t) for t in loss_member_eachbatch_poison]#get each batch avg member loss in each model
            loss_nonmember_eachbatch_avg_poison=[sum(t) / len(t) for t in loss_nomember_eachbatch_poison]#get each batch avg nonmember loss in each model
            loss_member_poison_list.append(loss_member_eachbatch_avg_poison)
            loss_nonmember_poison_list.append(loss_nonmember_eachbatch_avg_poison)

            diff_loss=[a-b for a,b in zip(loss_labal_ori_list,loss_labal_poison_list)]
            diff_grad=[a-b for a,b in zip(model_gradients,model_gradients_poison)]

            for i in range(len(diff_loss)):
                diff_loss[i]=diff_loss[i].to(self.device)
            for i in range(len(diff_grad)):
                diff_grad[i]=diff_grad[i].to(self.device)


            

            #print()
            #print()
            
            ####################################################

            ####################################################
            # qian 32 ge cheng yuan hou 32 ge fei cheng yuan
            #mean_member_loss = sum(member_loss) / len(member_loss)
            # print(len(nonmember_loss))

            #mean_nonmember_loss = sum(nonmember_loss) / len(nonmember_loss)
            # print(mean_member_loss)
            # print(mean_nonmember_loss)
            #loss_member_list.append(mean_member_loss)
            #loss_nonmember_list.append(mean_nonmember_loss)
            ####################################################

            # Get the predictions for the membership classification
            self.attack_model.to(self.device)
            
            member_input = Variable(member_input.data, requires_grad=True)
            non_member_input=Variable(non_member_input,requires_grad=True)

            member_input_poison=Variable(member_input.data, requires_grad=True)
            non_member_input_poison=Variable(non_member_input_poison,requires_grad=True)

            for i in range(len(diff_loss)):
                diff_loss[i]=Variable(diff_loss[i],requires_grad=True)

            for i in range(len(diff_grad)):
                diff_grad[i]=Variable(diff_grad[i],requires_grad=True)


            # Change labels of the data for binary attack classification
            member_target = torch.Tensor([1 for _ in member_input])
            non_member_target = torch.Tensor([0 for _ in non_member_input])
            membership_labels = torch.cat((member_target, non_member_target))
            membership_labels = torch.unsqueeze(membership_labels, -1).data.float().to(self.device)

            # Get membership predictions
            ###only model_gradients diffence
            membership_predictions = self.attack_model(model_outputs, one_hot_labels, loss_values, model_gradients,diff_loss,diff_grad)
            # Calculate the loss of attack model
            attack_loss = self.attack_loss_function(membership_predictions, membership_labels)

            # Perform optimizer steps
            attack_loss.backward(retain_graph=True)

            patch_grad_member = member_input.grad.clone().cpu()
            patch_grad_non_member = non_member_input.grad.clone().cpu()
            patch_grad_member_poison = member_input_poison.grad.clone().cpu()
            patch_grad_non_member_poison = non_member_input_poison.grad.clone().cpu()
            print(member_input[0].shape)
            print(non_member_input[0].shape)
            plt.imsave(member_input[0].cpu().detach().numpy().transpose(1,2,0),'./member.png')
            plt.imsave(non_member_input[0].cpu().datach().numpy().transpose(1,2,0),'./nonmember.png')
            print(patch_grad_member.shape)
            print(patch_grad_non_member.shape)
            #patch_grad_member.grad.data.zero_()
            applied_patch, mask, x_location, y_location = mask_generation(self.patch_type, self.adv_patch_cpu, image_size=(3,32,32))
            lr=0.01
            #grad img plus patch
            applied_patch = lr*patch_grad_member_poison+lr*patch_grad_non_member_poison+lr * patch_grad_member+lr*patch_grad_non_member + applied_patch.type(torch.FloatTensor)
            self.adv_patch_cpu=applied_patch
            print(applied_patch.shape)

            ###after backward() habe grad
            #self.attack_optimizer.step()

            # Place model back to CPU
            #self.attack_model.to('cpu')

            # Measure training accuracy and report metric
            batch_confusion_matrix.update(membership_predictions, membership_labels)
            accuracy.update(batch_confusion_matrix.get_accuracy(), self.attack_train_batch_size)
            losses.update(attack_loss.item(), self.attack_train_batch_size)

            temp = batch_confusion_matrix.get_confusion_matrix()
            final_confusion_matrix.update_from_matrix(temp)

            message = f'[ Round: {round_number} ' \
                      f'| Attacker Train ' \
                      f'| Batch Time: {time.time() - batch_start_time:.2f}s ' \
                      f'| Batch Loss: {attack_loss:.5f} ' \
                      f'| Batch Acc: {batch_confusion_matrix.get_accuracy():.2f}% ]' \
                      f'| Batch Conf Matrix: TP:{temp[0]}' \
                      f' FP:{temp[1]}' \
                      f' TN:{temp[2]}' \
                      f' FN:{temp[3]} ]'


            logging.info(message)
            if times%self.save_information_count==0:
                log_attack = logging.getLogger('log_attack')
                log_attack.info(message)
        
        '''

        l2_norm_member_round_avg=[sum(x) / len(x) for x in zip(*l2_norm_member_list)]
        l2_norm_nonmember_round_avg=[sum(x) / len(x) for x in zip(*l2_norm_nonmember_list)]
        loss_member_round_avg=[sum(x) / len(x) for x in zip(*loss_member_list)]
        loss_nonmember_round_avg=[sum(x) / len(x) for x in zip(*loss_nonmember_list)]
        decimal_places=4
        l2_norm_member_round_avg = [round(tensor.item(), decimal_places) for tensor in l2_norm_member_round_avg]
        l2_norm_nonmember_round_avg = [round(tensor.item(), decimal_places) for tensor in l2_norm_nonmember_round_avg]
        loss_member_round_avg = [round(tensor.item(), decimal_places) for tensor in loss_member_round_avg]
        loss_nonmember_round_avg = [round(tensor.item(), decimal_places) for tensor in loss_nonmember_round_avg]



        ####################################################

        temp = final_confusion_matrix.get_confusion_matrix()
        self.best_inference_acc_train = max(final_confusion_matrix.get_accuracy(),self.best_inference_acc_train)
        message = f'[ Round: {round_number} Totals ' \
                  f'| Attacker Train ' \
                  f'| Total Time: {time.time() - start_time:.2f}s ' \
                  f'| Avg Loss: {losses.avg:.5f} ' \
                  f'| Grad Norm Member: {l2_norm_member_round_avg} ' \
                  f'| Grad Norm NonMember: {l2_norm_nonmember_round_avg} ' \
                  f'| Loss Member: {loss_member_round_avg} ' \
                  f'| Loss NonMember: {loss_nonmember_round_avg} ' \
                  f'| Acc: {final_confusion_matrix.get_accuracy():.2f}% ]' \
                  f'| Final Conf Matrix: TP:{temp[0]}' \
                  f' FP:{temp[1]}' \
                  f' TN:{temp[2]}' \
                  f' FN:{temp[3]} ]'


        logging.info(message)
        log_attack = logging.getLogger('log_attack')
        log_attack.info(message)
    



    def train_attack2(self, round_number, target_models):
        """ Ja hier moet dus documentatie """

        # confusion matrix and accuracy metric initialized
        start_time = time.time()
        final_confusion_matrix = ConfusionMatrix()
        batch_confusion_matrix = ConfusionMatrix()
        losses = AverageMeter()
        accuracy = AverageMeter()
        # Set attack model to train
        self.attack_model.train()
        # Load in training data
        member, non_member = self.attack_train_data
        #############################################
        member_poison=self.attack_train_data_member_poison
        non_member_poison=self.attack_train_non_member_poison
        #############################
        '''
        if self.data_name == 'mnist' or self.data_name == 'fashion_mnist':
            dataset_poison_member = MyDataset_mnist(self.adv_patch_cpu, member, self.target, 1, mode="test")
            dataset_poison_nonmember = MyDataset_mnist(self.adv_patch_cpu, non_member, self.target, 1, mode="test")
        else:
            dataset_poison_member = MyDataset_cifar_new(self.adv_patch_cpu, member, self.target, 1, mode="test")
            dataset_poison_nonmember = MyDataset_cifar_new(self.adv_patch_cpu, non_member, self.target, 1, mode="test")
        '''


        #############################
        temp_data_loader = DataLoader(member,#member
                                      batch_size=self.attack_train_batch_size // 2,
                                      shuffle=False,
                                      num_workers=0,
                                      pin_memory=True
                                      )

        temp_data_loader_2 = DataLoader(non_member,#non_member
                                        batch_size=self.attack_train_batch_size // 2,
                                        shuffle=False,
                                        num_workers=0,
                                        pin_memory=True
                                        )

        temp_data_loader_poison=DataLoader(member_poison,
                                        batch_size=self.attack_train_batch_size // 2,
                                        shuffle=False,
                                        num_workers=0,
                                        pin_memory=True
        )


        temp_data_loader_2_poison=DataLoader(non_member_poison,
                                        batch_size=self.attack_train_batch_size // 2,
                                        shuffle=False,
                                        num_workers=0,
                                        pin_memory=True
        )


        data_loader = zip(temp_data_loader, temp_data_loader_2,temp_data_loader_poison,temp_data_loader_2_poison)
        #data_loader_poison=zip(temp_data_loader_poison,temp_data_loader_2_poison)
        ####################################################
        # bao cun mei ge batch de ti du fan shu he loss
        l2_norm_member_list = []
        l2_norm_nonmember_list = []
        loss_member_list = []
        loss_nonmember_list = []



        l2_norm_member_poison_list = []
        l2_norm_nonmember_poison_list = []
        loss_member_poison_list = []
        loss_nonmember_poison_list = []


        temp = self.batch_size
        halftbatch = temp // 2  # xu yao geng ju batch zhi tiaozheng!!!
        times = 0
        ####################################################
        for (member_input, member_target), (non_member_input, non_member_target),(member_input_poison, member_target_poison),(non_member_input_poison, non_member_target_poison) in data_loader:
            times += 1
            # if times==7:
            # print("round 7")
            # take timestamp
            batch_start_time = time.time()
            batch_confusion_matrix.reset()
            segmation = len(member_input)
            # Pre-define variables to use.
            model_outputs = []
            loss_values = []
            model_gradients = []



            model_outputs_poison = []
            loss_values_poison = []
            model_gradients_poison = []

            #temp_loss_values = []
            # Load data to device
            member_input, member_target = member_input.float().to(self.device) \
                , member_target.long().to(self.device)
            non_member_input, non_member_target = non_member_input.float().to(self.device) \
                , non_member_target.long().to(self.device)
            
            member_input_poison, member_target_poison = member_input_poison.float().to(self.device) \
                , member_target_poison.long().to(self.device)
            non_member_input_poison, non_member_target_poison = non_member_input_poison.float().to(self.device) \
                , non_member_target_poison.long().to(self.device)

            data, labels = torch.cat((member_input, non_member_input)), torch.cat(
                (member_target, non_member_target))

            # Create one-hot encoding of labels, as this has to be done only once.
            one_hot_labels = encode_labels(labels, self.one_hot_encoding, self.device).float()

            data_poison, labels_poison = torch.cat((member_input_poison, non_member_input_poison)), torch.cat(
                (member_target_poison, non_member_target_poison))
            #plt.imshow(data_poison[0].cpu().numpy().transpose(1,2,0))
            #plt.show()
            one_hot_labels_poison = encode_labels(labels_poison, self.one_hot_encoding, self.device).float()
            ####################
            loss_member_eachbatch=[]
            loss_nomember_eachbatch=[]

            loss_member_eachbatch_poison=[]
            loss_nomember_eachbatch_poison=[]
            ####################
            # For each observed model collect input data for the attack model.

            loss_labal_ori_list=[]

            imglist=[]
            for model in target_models:
                #member_input_list.append(member_input)
                #non_member_input_list.append(non_member_input)
                tmppp=torch.cat((member_input, non_member_input), dim=0)
                tmppp = Variable(tmppp, requires_grad=True)
                imglist.append(tmppp)
                #
                ####################################################
                # qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                member_loss = []
                nonmember_loss = []
                ####################################################
                model.eval()
                model.to(self.device)
                optimizer = optimizers.__dict__[self.optimizer_name](
                    params=model.parameters(),
                    lr=self.learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay
                )

                # Get predictions
                predictions = model(data)
                temp_pred = torch.from_numpy(predictions.data.cpu().numpy()).to(self.device)
                model_outputs.append(temp_pred.requires_grad_(True))  # ba cheng yuan he fei chen yuan
                # calculate loss values
                # predictions 64*10 one_hot_labels 64*10
                loss_value = torch.sum(predictions * one_hot_labels, dim=1).view([-1, 1])
                ###loss_value是對應標籤的模型輸出值
                temp_loss = torch.from_numpy(loss_value.data.cpu().numpy()).to(self.device)

                loss_values.append(temp_loss.requires_grad_(True))
                loss_labal=[]
                gradients = torch.zeros(0)
                # dui mei yige yangebn qiu loss he tidu
                # qian 32 ge shi cheng yaun hou 32 ge fei cheng yuan
                for index in range(predictions.size(0)):
                    # ji suan yi ge shi ti de loss, qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                    loss = self.loss_function(predictions[index].view([1, -1]), labels[index].view([-1]))
                    loss_labal.append(loss)
                    ################## add loss into list
                    if index < segmation:
                        member_loss.append(loss)
                    else:
                        nonmember_loss.append(loss)
                    ##################
                    optimizer.zero_grad()
                    if index == (predictions.size(0)) - 1:
                        loss.backward(retain_graph=False)
                    else:
                        loss.backward(retain_graph=True)
                    # current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])#qu zui hou yi ceng can shu de tidu
                    if self.model_name == "AlexNet":
                        current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name == "CNNCifar10":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    elif self.model_name == "CNNFashion_Mnist":
                        current_grads = model.fc.weight.grad.view([1, 1, 7 * 7 * 32, self.number_of_classes])
                    elif self.model_name == "CNNMnist":
                        current_grads = model.fc2.weight.grad.view([1, 1, 50, self.number_of_classes])
                    elif self.model_name == "LeNet5":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    # current_grads = model.lastlayer_name.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name=="CNNModel":
                        current_grads = model.fc2.weight.grad.view([1, 1, 256, self.number_of_classes])
                    else:
                        current_grads = None
                    if gradients.size()[0] == 0:
                        gradients = current_grads
                    else:
                        gradients = torch.cat((gradients, current_grads))
                # 64, 1, 256, 100 qian 32 ge shi chengyuan zuihouyicengde tidu hou 32shi fei cheng yuan de tidu
                temp_gradients = torch.from_numpy(gradients.data.cpu().numpy()).to(self.device)
                loss_labal_ori_list.append(torch.unsqueeze(torch.tensor(loss_labal), dim=1))

                model_gradients.append(temp_gradients.requires_grad_(True))
                loss_member_eachbatch.append(member_loss)
                loss_nomember_eachbatch.append(nonmember_loss)
                model.to('cpu')
            



            loss_labal_poison_list=[]
            for model in target_models:
                #
                ####################################################
                # qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                member_loss_poison = []
                nonmember_loss_poison = []
                ####################################################
                model.eval()
                model.to(self.device)
                optimizer = optimizers.__dict__[self.optimizer_name](
                    params=model.parameters(),
                    lr=self.learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay
                )

                # Get predictions
                predictions = model(data_poison)
                _, predicted = torch.max(predictions, 1)

                temp_pred = torch.from_numpy(predictions.data.cpu().numpy()).to(self.device)
                model_outputs_poison.append(temp_pred.requires_grad_(True))  # ba cheng yuan he fei chen yuan
                # calculate loss values
                # predictions 64*10 one_hot_labels 64*10
                loss_value = torch.sum(predictions * one_hot_labels_poison, dim=1).view([-1, 1])

                ###loss_value是對應標籤的模型輸出值
                temp_loss = torch.from_numpy(loss_value.data.cpu().numpy()).to(self.device)

                loss_values_poison.append(temp_loss.requires_grad_(True))

                gradients = torch.zeros(0)
                # dui mei yige yangebn qiu loss he tidu
                # qian 32 ge shi cheng yaun hou 32 ge fei cheng yuan
                loss_labal=[]
                for index in range(predictions.size(0)):
                    # ji suan yi ge shi ti de loss, qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                    loss = self.loss_function(predictions[index].view([1, -1]), labels[index].view([-1]))
                    loss_labal.append(loss)
                    ################## add loss into list
                    if index < segmation:
                        member_loss_poison.append(loss)
                    else:
                        nonmember_loss_poison.append(loss)
                    ##################
                    optimizer.zero_grad()
                    if index == (predictions.size(0)) - 1:
                        loss.backward(retain_graph=False)
                    else:
                        loss.backward(retain_graph=True)
                    # current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])#qu zui hou yi ceng can shu de tidu
                    if self.model_name == "AlexNet":
                        current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name == "CNNCifar10":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    elif self.model_name == "CNNFashion_Mnist":
                        current_grads = model.fc.weight.grad.view([1, 1, 7 * 7 * 32, self.number_of_classes])
                    elif self.model_name == "CNNMnist":
                        current_grads = model.fc2.weight.grad.view([1, 1, 50, self.number_of_classes])
                    elif self.model_name == "LeNet5":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    # current_grads = model.lastlayer_name.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name=="CNNModel":
                        current_grads = model.fc2.weight.grad.view([1, 1, 256, self.number_of_classes])
                    else:
                        current_grads = None
                    if gradients.size()[0] == 0:
                        gradients = current_grads
                    else:
                        gradients = torch.cat((gradients, current_grads))
                # 64, 1, 256, 100 qian 32 ge shi chengyuan zuihouyicengde tidu hou 32shi fei cheng yuan de tidu
                temp_gradients = torch.from_numpy(gradients.data.cpu().numpy()).to(self.device)
                loss_labal_poison_list.append(torch.unsqueeze(torch.tensor(loss_labal), dim=1))
                model_gradients_poison.append(temp_gradients.requires_grad_(True))
                loss_member_eachbatch_poison.append(member_loss)
                loss_nomember_eachbatch_poison.append(nonmember_loss)
                model.to('cpu')



            ####################################################
            ##########ji suan yi ge batch de tidu fan shu
            norm_member_eachbatch = []
            norm_nomember_eachbatch = []

            norm_member_eachbatch_poison = []
            norm_nomember_eachbatch_poison = []

            for model_lastlayer_grads in model_gradients:
                model_member_grad = model_lastlayer_grads[:segmation]  # qu 0-31

                model_nonmember_grad = model_lastlayer_grads[segmation:]  # qu 32-63
                l2_norm_member = torch.norm(model_member_grad, p=2)
                #for tmp_member in model_member_grad:
                #    print('l2norm member:',torch.norm(tmp_member, p=2))
                #for tmp_non_member in model_nonmember_grad:
                #    print('l2norm non member:',torch.norm(tmp_non_member, p=2))
                l2_norm_nonmember = torch.norm(model_nonmember_grad, p=2)
                # fang dao liang ge lie biao
                # fen bie bao cun mei yi batch de ti du fan shu
                norm_member_eachbatch.append(l2_norm_member)
                norm_nomember_eachbatch.append(l2_norm_nonmember)
            l2_norm_member_list.append(norm_member_eachbatch)
            l2_norm_nonmember_list.append(norm_nomember_eachbatch)


            for model_lastlayer_grads in model_gradients_poison:
                model_member_grad = model_lastlayer_grads[:segmation]  # qu 0-31

                model_nonmember_grad = model_lastlayer_grads[segmation:]  # qu 32-63
                l2_norm_member = torch.norm(model_member_grad, p=2)
                #for tmp_member in model_member_grad:
                #    print('l2norm member:',torch.norm(tmp_member, p=2))
                #for tmp_non_member in model_nonmember_grad:
                #    print('l2norm non member:',torch.norm(tmp_non_member, p=2))
                l2_norm_nonmember = torch.norm(model_nonmember_grad, p=2)
                # fang dao liang ge lie biao
                # fen bie bao cun mei yi batch de ti du fan shu
                norm_member_eachbatch_poison.append(l2_norm_member)
                norm_nomember_eachbatch_poison.append(l2_norm_nonmember)
            l2_norm_member_poison_list.append(loss_member_eachbatch_poison)
            l2_norm_nonmember_poison_list.append(norm_nomember_eachbatch_poison)            

            loss_member_eachbatch_avg=[sum(t) / len(t) for t in loss_member_eachbatch]#get each batch avg member loss in each model
            loss_nonmember_eachbatch_avg=[sum(t) / len(t) for t in loss_nomember_eachbatch]#get each batch avg nonmember loss in each model
            loss_member_list.append(loss_member_eachbatch_avg)
            loss_nonmember_list.append(loss_nonmember_eachbatch_avg)


            loss_member_eachbatch_avg_poison=[sum(t) / len(t) for t in loss_member_eachbatch_poison]#get each batch avg member loss in each model
            loss_nonmember_eachbatch_avg_poison=[sum(t) / len(t) for t in loss_nomember_eachbatch_poison]#get each batch avg nonmember loss in each model
            loss_member_poison_list.append(loss_member_eachbatch_avg_poison)
            loss_nonmember_poison_list.append(loss_nonmember_eachbatch_avg_poison)

            diff_loss=[a-b for a,b in zip(loss_labal_ori_list,loss_labal_poison_list)]
            diff_grad=[a-b for a,b in zip(model_gradients,model_gradients_poison)]

            for i in range(len(diff_loss)):
                diff_loss[i]=diff_loss[i].to(self.device)
            for i in range(len(diff_grad)):
                diff_grad[i]=diff_grad[i].to(self.device)


            

            #print()
            #print()
            
            ####################################################

            ####################################################
            # qian 32 ge cheng yuan hou 32 ge fei cheng yuan
            #mean_member_loss = sum(member_loss) / len(member_loss)
            # print(len(nonmember_loss))

            #mean_nonmember_loss = sum(nonmember_loss) / len(nonmember_loss)
            # print(mean_member_loss)
            # print(mean_nonmember_loss)
            #loss_member_list.append(mean_member_loss)
            #loss_nonmember_list.append(mean_nonmember_loss)
            ####################################################

            # Get the predictions for the membership classification
            self.attack_model.to(self.device)
            self.attack_optimizer.zero_grad()

            # Change labels of the data for binary attack classification
            member_target = torch.Tensor([1 for _ in member_input])
            non_member_target = torch.Tensor([0 for _ in non_member_input])
            membership_labels = torch.cat((member_target, non_member_target))
            membership_labels = torch.unsqueeze(membership_labels, -1).data.float().to(self.device)

            # Get membership predictions
            ###only model_gradients diffence

            


            membership_predictions = self.attack_model(model_outputs, one_hot_labels, loss_values, model_gradients,diff_loss,diff_grad,imglist)
            # Calculate the loss of attack model
            attack_loss = self.attack_loss_function(membership_predictions, membership_labels)

            # Perform optimizer steps
            attack_loss.backward()
            ###after backward() habe grad
            self.attack_optimizer.step()



            # Place model back to CPU
            self.attack_model.to('cpu')


            

            # Measure training accuracy and report metric
            batch_confusion_matrix.update(membership_predictions, membership_labels)
            accuracy.update(batch_confusion_matrix.get_accuracy(), self.attack_train_batch_size)
            losses.update(attack_loss.item(), self.attack_train_batch_size)

            temp = batch_confusion_matrix.get_confusion_matrix()
            final_confusion_matrix.update_from_matrix(temp)

            message = f'[ Round: {round_number} ' \
                      f'| Attacker Train ' \
                      f'| Batch Time: {time.time() - batch_start_time:.2f}s ' \
                      f'| Batch Loss: {attack_loss:.5f} ' \
                      f'| Batch Acc: {batch_confusion_matrix.get_accuracy():.2f}% ]' \
                      f'| Batch Conf Matrix: TP:{temp[0]}' \
                      f' FP:{temp[1]}' \
                      f' TN:{temp[2]}' \
                      f' FN:{temp[3]} ]'


            logging.info(message)
            if times%self.save_information_count==0:
                log_attack = logging.getLogger('log_attack')
                log_attack.info(message)
        

        l2_norm_member_round_avg=[sum(x) / len(x) for x in zip(*l2_norm_member_list)]
        l2_norm_nonmember_round_avg=[sum(x) / len(x) for x in zip(*l2_norm_nonmember_list)]
        loss_member_round_avg=[sum(x) / len(x) for x in zip(*loss_member_list)]
        loss_nonmember_round_avg=[sum(x) / len(x) for x in zip(*loss_nonmember_list)]
        decimal_places=4
        l2_norm_member_round_avg = [round(tensor.item(), decimal_places) for tensor in l2_norm_member_round_avg]
        l2_norm_nonmember_round_avg = [round(tensor.item(), decimal_places) for tensor in l2_norm_nonmember_round_avg]
        loss_member_round_avg = [round(tensor.item(), decimal_places) for tensor in loss_member_round_avg]
        loss_nonmember_round_avg = [round(tensor.item(), decimal_places) for tensor in loss_nonmember_round_avg]



        ####################################################

        temp = final_confusion_matrix.get_confusion_matrix()
        self.best_inference_acc_train = max(final_confusion_matrix.get_accuracy(),self.best_inference_acc_train)
        message = f'[ Round: {round_number} Totals ' \
                  f'| Attacker Train ' \
                  f'| Total Time: {time.time() - start_time:.2f}s ' \
                  f'| Avg Loss: {losses.avg:.5f} ' \
                  f'| Grad Norm Member: {l2_norm_member_round_avg} ' \
                  f'| Grad Norm NonMember: {l2_norm_nonmember_round_avg} ' \
                  f'| Loss Member: {loss_member_round_avg} ' \
                  f'| Loss NonMember: {loss_nonmember_round_avg} ' \
                  f'| Acc: {final_confusion_matrix.get_accuracy():.2f}% ]' \
                  f'| Final Conf Matrix: TP:{temp[0]}' \
                  f' FP:{temp[1]}' \
                  f' TN:{temp[2]}' \
                  f' FN:{temp[3]} ]'


        logging.info(message)
        log_attack = logging.getLogger('log_attack')
        log_attack.info(message)


    def train_trigger(self, round_number, target_models,lr=0.01,way=1):
        # Metrics
        start_time = time.time()
        final_confusion_matrix = ConfusionMatrix()
        batch_confusion_matrix = ConfusionMatrix()
        losses = AverageMeter()
        accuracy = AverageMeter()
        # create dict for keeping track of scores per class.
        class_matrices = dict((data_class, ConfusionMatrix()) for data_class in range(self.number_of_classes))

        self.attack_model.eval()

        member, non_member = self.attack_test_data
        temp_data_loader = DataLoader(member,
                                    batch_size=self.attack_test_batch_size // 2,
                                    shuffle=False,
                                    num_workers=2,
                                    pin_memory=True
                                    )

        temp_data_loader_2 = DataLoader(non_member,
                                        batch_size=self.attack_test_batch_size // 2,
                                        shuffle=False,
                                        num_workers=2,
                                        pin_memory=True
                                        )
        
        member_poison=self.attack_test_member_poison
        non_member_poison=self.attack_test_non_member_poison

        temp_data_loader_poison=DataLoader(member_poison,
                                        batch_size=self.attack_train_batch_size // 2,
                                        shuffle=False,
                                        num_workers=0,
                                        pin_memory=True
        )


        temp_data_loader_2_poison=DataLoader(non_member_poison,
                                        batch_size=self.attack_train_batch_size // 2,
                                        shuffle=False,
                                        num_workers=0,
                                        pin_memory=True
        )

        data_loader = zip(temp_data_loader, temp_data_loader_2,temp_data_loader_poison,temp_data_loader_2_poison)
        #data_loader_poison=zip(temp_data_loader_poison,temp_data_loader_2_poison)
        ####################################################
        # bao cun mei ge batch de ti du fan shu he loss
        l2_norm_member_list = []
        l2_norm_nonmember_list = []
        loss_member_list = []
        loss_nonmember_list = []



        l2_norm_member_poison_list = []
        l2_norm_nonmember_poison_list = []
        loss_member_poison_list = []
        loss_nonmember_poison_list = []


        temp = self.batch_size
        halftbatch = temp // 2  # xu yao geng ju batch zhi tiaozheng!!!
        times = 0
        ####################################################
        for (member_input, member_target), (non_member_input, non_member_target),(member_input_poison, member_target_poison),(non_member_input_poison, non_member_target_poison) in data_loader:
            times+=1
            #
            batch_start_time = time.time()
            batch_confusion_matrix.reset()
            segmation = len(member_input)
            # Pre-define variables to use.
            model_outputs = []
            loss_values = []
            model_gradients = []



            model_outputs_poison = []
            loss_values_poison = []
            model_gradients_poison = []

            #temp_loss_values = []
            # Load data to device
            member_input, member_target = member_input.float().to(self.device) \
                , member_target.long().to(self.device)
            non_member_input, non_member_target = non_member_input.float().to(self.device) \
                , non_member_target.long().to(self.device)
            
            member_input_poison, member_target_poison = member_input_poison.float().to(self.device) \
                , member_target_poison.long().to(self.device)
            non_member_input_poison, non_member_target_poison = non_member_input_poison.float().to(self.device) \
                , non_member_target_poison.long().to(self.device)

            data, labels = torch.cat((member_input, non_member_input)), torch.cat(
                (member_target, non_member_target))

            # Create one-hot encoding of labels, as this has to be done only once.
            one_hot_labels = encode_labels(labels, self.one_hot_encoding, self.device).float()

            data_poison, labels_poison = torch.cat((member_input_poison, non_member_input_poison)), torch.cat(
                (member_target_poison, non_member_target_poison))
            #plt.imshow(data_poison[0].cpu().numpy().transpose(1,2,0))
            #plt.show()
            one_hot_labels_poison = encode_labels(labels_poison, self.one_hot_encoding, self.device).float()
            ####################
            loss_member_eachbatch=[]
            loss_nomember_eachbatch=[]

            loss_member_eachbatch_poison=[]
            loss_nomember_eachbatch_poison=[]
            ####################
            # For each observed model collect input data for the attack model.

            loss_labal_ori_list=[]

           
            imglist=[]
            for model in target_models:
                #member_input_list.append(member_input)
                #non_member_input_list.append(non_member_input)
                tmppp=torch.cat((member_input, non_member_input), dim=0)
                tmppp = Variable(tmppp, requires_grad=True)
                imglist.append(tmppp)



                #
                ####################################################
                # qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                member_loss = []
                nonmember_loss = []
                ####################################################
                model.eval()
                model.to(self.device)
                optimizer = optimizers.__dict__[self.optimizer_name](
                    params=model.parameters(),
                    lr=self.learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay
                )

                # Get predictions
                predictions = model(data)
                temp_pred = torch.from_numpy(predictions.data.cpu().numpy()).to(self.device)
                model_outputs.append(temp_pred.requires_grad_(True))  # ba cheng yuan he fei chen yuan
                # calculate loss values
                # predictions 64*10 one_hot_labels 64*10
                loss_value = torch.sum(predictions * one_hot_labels, dim=1).view([-1, 1])
                ###loss_value是對應標籤的模型輸出值
                temp_loss = torch.from_numpy(loss_value.data.cpu().numpy()).to(self.device)

                loss_values.append(temp_loss.requires_grad_(True))
                loss_labal=[]
                gradients = torch.zeros(0)
                # dui mei yige yangebn qiu loss he tidu
                # qian 32 ge shi cheng yaun hou 32 ge fei cheng yuan
                for index in range(predictions.size(0)):
                    # ji suan yi ge shi ti de loss, qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                    loss = self.loss_function(predictions[index].view([1, -1]), labels[index].view([-1]))
                    loss_labal.append(loss)
                    ################## add loss into list
                    if index < segmation:
                        member_loss.append(loss)
                    else:
                        nonmember_loss.append(loss)
                    ##################
                    optimizer.zero_grad()
                    if index == (predictions.size(0)) - 1:
                        loss.backward(retain_graph=False)
                    else:
                        loss.backward(retain_graph=True)
                    # current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])#qu zui hou yi ceng can shu de tidu
                    if self.model_name == "AlexNet":
                        current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name == "CNNCifar10":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    elif self.model_name == "CNNFashion_Mnist":
                        current_grads = model.fc.weight.grad.view([1, 1, 7 * 7 * 32, self.number_of_classes])
                    elif self.model_name == "CNNMnist":
                        current_grads = model.fc2.weight.grad.view([1, 1, 50, self.number_of_classes])
                    elif self.model_name == "LeNet5":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    # current_grads = model.lastlayer_name.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name=="CNNModel":
                        current_grads = model.fc2.weight.grad.view([1, 1, 256, self.number_of_classes])
                    else:
                        current_grads = None
                    if gradients.size()[0] == 0:
                        gradients = current_grads
                    else:
                        gradients = torch.cat((gradients, current_grads))
                # 64, 1, 256, 100 qian 32 ge shi chengyuan zuihouyicengde tidu hou 32shi fei cheng yuan de tidu
                temp_gradients = torch.from_numpy(gradients.data.cpu().numpy()).to(self.device)
                loss_labal_ori_list.append(torch.unsqueeze(torch.tensor(loss_labal), dim=1))

                model_gradients.append(temp_gradients.requires_grad_(True))
                loss_member_eachbatch.append(member_loss)
                loss_nomember_eachbatch.append(nonmember_loss)
                model.to('cpu')
            



            loss_labal_poison_list=[]
            for model in target_models:
                #
                ####################################################
                # qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                member_loss_poison = []
                nonmember_loss_poison = []
                ####################################################
                model.eval()
                model.to(self.device)
                optimizer = optimizers.__dict__[self.optimizer_name](
                    params=model.parameters(),
                    lr=self.learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay
                )

                # Get predictions
                predictions = model(data_poison)
                _, predicted = torch.max(predictions, 1)

                temp_pred = torch.from_numpy(predictions.data.cpu().numpy()).to(self.device)
                model_outputs_poison.append(temp_pred.requires_grad_(True))  # ba cheng yuan he fei chen yuan
                # calculate loss values
                # predictions 64*10 one_hot_labels 64*10
                loss_value = torch.sum(predictions * one_hot_labels_poison, dim=1).view([-1, 1])

                ###loss_value是對應標籤的模型輸出值
                temp_loss = torch.from_numpy(loss_value.data.cpu().numpy()).to(self.device)

                loss_values_poison.append(temp_loss.requires_grad_(True))

                gradients = torch.zeros(0)
                # dui mei yige yangebn qiu loss he tidu
                # qian 32 ge shi cheng yaun hou 32 ge fei cheng yuan
                loss_labal=[]
                for index in range(predictions.size(0)):
                    # ji suan yi ge shi ti de loss, qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                    loss = self.loss_function(predictions[index].view([1, -1]), labels[index].view([-1]))
                    loss_labal.append(loss)
                    ################## add loss into list
                    if index < segmation:
                        member_loss_poison.append(loss)
                    else:
                        nonmember_loss_poison.append(loss)
                    ##################
                    optimizer.zero_grad()
                    if index == (predictions.size(0)) - 1:
                        loss.backward(retain_graph=False)
                    else:
                        loss.backward(retain_graph=True)
                    # current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])#qu zui hou yi ceng can shu de tidu
                    if self.model_name == "AlexNet":
                        current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name == "CNNCifar10":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    elif self.model_name == "CNNFashion_Mnist":
                        current_grads = model.fc.weight.grad.view([1, 1, 7 * 7 * 32, self.number_of_classes])
                    elif self.model_name == "CNNMnist":
                        current_grads = model.fc2.weight.grad.view([1, 1, 50, self.number_of_classes])
                    elif self.model_name == "LeNet5":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    # current_grads = model.lastlayer_name.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name=="CNNModel":
                        current_grads = model.fc2.weight.grad.view([1, 1, 256, self.number_of_classes])
                    else:
                        current_grads = None
                    if gradients.size()[0] == 0:
                        gradients = current_grads
                    else:
                        gradients = torch.cat((gradients, current_grads))
                # 64, 1, 256, 100 qian 32 ge shi chengyuan zuihouyicengde tidu hou 32shi fei cheng yuan de tidu
                temp_gradients = torch.from_numpy(gradients.data.cpu().numpy()).to(self.device)
                loss_labal_poison_list.append(torch.unsqueeze(torch.tensor(loss_labal), dim=1))
                model_gradients_poison.append(temp_gradients.requires_grad_(True))
                loss_member_eachbatch_poison.append(member_loss)
                loss_nomember_eachbatch_poison.append(nonmember_loss)
                model.to('cpu')



            ####################################################
            ##########ji suan yi ge batch de tidu fan shu
            norm_member_eachbatch = []
            norm_nomember_eachbatch = []

            norm_member_eachbatch_poison = []
            norm_nomember_eachbatch_poison = []

            for model_lastlayer_grads in model_gradients:
                model_member_grad = model_lastlayer_grads[:segmation]  # qu 0-31

                model_nonmember_grad = model_lastlayer_grads[segmation:]  # qu 32-63
                l2_norm_member = torch.norm(model_member_grad, p=2)
                #for tmp_member in model_member_grad:
                #    print('l2norm member:',torch.norm(tmp_member, p=2))
                #for tmp_non_member in model_nonmember_grad:
                #    print('l2norm non member:',torch.norm(tmp_non_member, p=2))
                l2_norm_nonmember = torch.norm(model_nonmember_grad, p=2)
                # fang dao liang ge lie biao
                # fen bie bao cun mei yi batch de ti du fan shu
                norm_member_eachbatch.append(l2_norm_member)
                norm_nomember_eachbatch.append(l2_norm_nonmember)
            l2_norm_member_list.append(norm_member_eachbatch)
            l2_norm_nonmember_list.append(norm_nomember_eachbatch)


            for model_lastlayer_grads in model_gradients_poison:
                model_member_grad = model_lastlayer_grads[:segmation]  # qu 0-31

                model_nonmember_grad = model_lastlayer_grads[segmation:]  # qu 32-63
                l2_norm_member = torch.norm(model_member_grad, p=2)
                #for tmp_member in model_member_grad:
                #    print('l2norm member:',torch.norm(tmp_member, p=2))
                #for tmp_non_member in model_nonmember_grad:
                #    print('l2norm non member:',torch.norm(tmp_non_member, p=2))
                l2_norm_nonmember = torch.norm(model_nonmember_grad, p=2)
                # fang dao liang ge lie biao
                # fen bie bao cun mei yi batch de ti du fan shu
                norm_member_eachbatch_poison.append(l2_norm_member)
                norm_nomember_eachbatch_poison.append(l2_norm_nonmember)
            l2_norm_member_poison_list.append(loss_member_eachbatch_poison)
            l2_norm_nonmember_poison_list.append(norm_nomember_eachbatch_poison)            

            loss_member_eachbatch_avg=[sum(t) / len(t) for t in loss_member_eachbatch]#get each batch avg member loss in each model
            loss_nonmember_eachbatch_avg=[sum(t) / len(t) for t in loss_nomember_eachbatch]#get each batch avg nonmember loss in each model
            loss_member_list.append(loss_member_eachbatch_avg)
            loss_nonmember_list.append(loss_nonmember_eachbatch_avg)


            loss_member_eachbatch_avg_poison=[sum(t) / len(t) for t in loss_member_eachbatch_poison]#get each batch avg member loss in each model
            loss_nonmember_eachbatch_avg_poison=[sum(t) / len(t) for t in loss_nomember_eachbatch_poison]#get each batch avg nonmember loss in each model
            loss_member_poison_list.append(loss_member_eachbatch_avg_poison)
            loss_nonmember_poison_list.append(loss_nonmember_eachbatch_avg_poison)

            diff_loss=[a-b for a,b in zip(loss_labal_ori_list,loss_labal_poison_list)]
            diff_grad=[a-b for a,b in zip(model_gradients,model_gradients_poison)]

            for i in range(len(diff_loss)):
                diff_loss[i]=diff_loss[i].to(self.device)
            for i in range(len(diff_grad)):
                diff_grad[i]=diff_grad[i].to(self.device)

            # Get the predictions for the membership classification
            self.attack_model.to(self.device)

            # Get membership predictions
            membership_predictions = self.attack_model(model_outputs, one_hot_labels, loss_values,
                                                    model_gradients,diff_loss,diff_grad,imglist)

            # Change labels of the data for binary attack classification
            member_target = torch.Tensor([1 for _ in member_target])
            non_member_target = torch.Tensor([0 for _ in non_member_target])
            membership_labels = torch.cat((member_target, non_member_target))
            membership_labels = torch.unsqueeze(membership_labels, -1).data.float().to(self.device)

            # Calculate the loss of attack model
            attack_loss = self.attack_loss_function(membership_predictions, membership_labels)
            attack_loss.backward()

            applied_patch=self.adv_patch_cpu

            tempgrad=imglist[0].grad
            for i in range(1,len(imglist)):
                tempgrad=imglist[i].grad
        
            sum_result = torch.sum(tempgrad, dim=0)




            patch_grad = sum_result.clone().cpu()
            patch_grad=patch_grad[:,0:applied_patch.shape[1],0:applied_patch.shape[2]]

            #grad img plus patch
            #applied_patch = lr * patch_grad + applied_patch
            #print(type(applied_patch))
            #print(type(patch_grad))

            applied_patch_np = applied_patch.numpy() if isinstance(applied_patch, torch.Tensor) else applied_patch
            patch_grad_np=patch_grad.numpy() if isinstance(patch_grad, torch.Tensor) else patch_grad
            if way==1:
                applied_patch = applied_patch_np-lr * patch_grad_np
            else:
                applied_patch = applied_patch_np+lr * patch_grad_np
                 

            self.adv_patch_cpu=applied_patch
            

            # update class confusion matrices, to see scores per original class
            combined_labels = zip(labels, membership_labels)
            for index, (data_class, member_class) in enumerate(combined_labels):
                class_matrices[data_class.item()].update(membership_predictions[index], member_class)

            batch_confusion_matrix.update(membership_predictions, membership_labels)
            accuracy.update(batch_confusion_matrix.get_accuracy(), self.attack_test_batch_size)
            losses.update(attack_loss.item(), self.attack_test_batch_size)

            self.attack_model.to('cpu')


    def regenpoisondata(self):
        distribution = self.attack_data_distribution
        train_member_length = distribution[0]
        train_non_member_length = distribution[1]
        test_member_length = distribution[2]
        test_non_member_length = distribution[3]
        if self.data_name=='cifar10':
            attack_train_member_path = './cifar10data/attack_train_member_' + str(train_member_length) + '.pth'
            attack_train_nonmember_path = './cifar10data/attack_train_nonmember_' + str(
                train_non_member_length) + '.pth'
            attack_test_member_path = './cifar10data/attack_test_member_' + str(test_member_length) + '.pth'
            attack_test_nonmember_path = './cifar10data/attack_test_nonmember_' + str(test_non_member_length) + '.pth'
        elif self.data_name=='mnist':
            attack_train_member_path = './mnistdata/attack_train_member_' + str(train_member_length) + '.pth'
            attack_train_nonmember_path = './mnistdata/attack_train_nonmember_' + str(
                train_non_member_length) + '.pth'
            attack_test_member_path = './mnistdata/attack_test_member_' + str(test_member_length) + '.pth'
            attack_test_nonmember_path = './mnistdata/attack_test_nonmember_' + str(test_non_member_length) + '.pth'
        elif self.data_name == 'fashion_mnist':
            attack_train_member_path = './fashion_mnistdata/attack_train_member_' + str(train_member_length) + '.pth'
            attack_train_nonmember_path = './fashion_mnistdata/attack_train_nonmember_' + str(
                train_non_member_length) + '.pth'
            attack_test_member_path = './fashion_mnistdata/attack_test_member_' + str(test_member_length) + '.pth'
            attack_test_nonmember_path = './fashion_mnistdata/attack_test_nonmember_' + str(test_non_member_length) + '.pth'
        else:
            attack_train_member_path = './cifar100data/attack_train_member_' + str(train_member_length) + '.pth'
            attack_train_nonmember_path = './cifar100data/attack_train_nonmember_' + str(train_non_member_length) + '.pth'
            attack_test_member_path = './cifar100data/attack_test_member_' + str(test_member_length) + '.pth'
            attack_test_nonmember_path = './cifar100data/attack_test_nonmember_' + str(test_non_member_length) + '.pth'

        
        train_member = torch.load(attack_train_member_path)
        train_non_member = torch.load(attack_train_nonmember_path)
        test_member = torch.load(attack_test_member_path)
        test_non_member = torch.load(attack_test_nonmember_path)



        if self.data_name == 'mnist' or self.data_name == 'fashion_mnist':
            #train_member = MyDataset_mnist(self.adv_patch_cpu, train_member, self.target, self.test_back_prop, mode="train")
            #train_non_member = MyDataset_mnist(self.adv_patch_cpu, train_non_member, self.target, self.test_back_prop, mode="train")
            #test_member = MyDataset_mnist(self.adv_patch_cpu, test_member, self.target, self.test_back_prop, mode="train")
            #test_non_member = MyDataset_mnist(self.adv_patch_cpu, test_non_member, self.target, self.test_back_prop, mode="train")
            
            train_member_poison = MyDataset_mnist(self.adv_patch_cpu, train_member,
                                                self.target, self.back_prop, mode="train")
            
            train_non_member_poison=MyDataset_mnist(self.adv_patch_cpu, train_non_member,
                                                self.target, self.back_prop, mode="train")

            test_member_poison=MyDataset_mnist(self.adv_patch_cpu, test_member,
                                                self.target, self.back_prop, mode="train")

            test_non_member_poison=MyDataset_mnist(self.adv_patch_cpu, test_non_member,
                                                self.target, self.back_prop, mode="train")
        else:
            train_member_poison = MyDataset_cifar_new(self.adv_patch_cpu, train_member,
                                                self.target, self.back_prop, mode="train",
                                                number_of_classes=self.number_of_classes)
            train_non_member_poison=MyDataset_cifar_new(self.adv_patch_cpu, train_non_member,
                                                self.target, self.back_prop, mode="train",
                                                number_of_classes=self.number_of_classes)

            test_member_poison=MyDataset_cifar_new(self.adv_patch_cpu, test_member,
                                                self.target, self.back_prop, mode="train",
                                                number_of_classes=self.number_of_classes)
            test_non_member_poison=MyDataset_cifar_new(self.adv_patch_cpu, test_non_member,
                                                self.target, self.back_prop, mode="train",
                                            number_of_classes=self.number_of_classes)

    
        self.attack_train_data_member_poison = train_member_poison
        self.attack_train_non_member_poison = train_non_member_poison

        self.attack_test_member_poison=test_member_poison
        self.attack_test_non_member_poison=test_non_member_poison



    def plot_tsne(self,target_models,model_path):
        features_in_hook = []
        features_out_hook = []



        def hook(module, fea_in, fea_out):
            features_in_hook.append(fea_in)
            features_out_hook.append(fea_out)

        for layer in self.attack_model.modules():
            #if isinstance(layer, nn.Linear):
            if isinstance(layer, nn.ReLU):
                # print(layer)
                layer.register_forward_hook(hook=hook)

        start_time = time.time()
        final_confusion_matrix = ConfusionMatrix()
        batch_confusion_matrix = ConfusionMatrix()
        losses = AverageMeter()
        accuracy = AverageMeter()
        # Set attack model to train
        self.attack_model.train()
        # Load in training data
        member, non_member = self.attack_test_data
        encoder_outputs=[]
        temp_data_loader = DataLoader(member,#member
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=0,
                                      pin_memory=True
                                      )

        temp_data_loader_2 = DataLoader(non_member,#non_member
                                        batch_size=1,
                                        shuffle=True,
                                        num_workers=0,
                                        pin_memory=True
                                        )

        ####################################################

        times = 0
        for member_input, member_target in tqdm(temp_data_loader):
            times += 1

            batch_confusion_matrix.reset()

            # Pre-define variables to use.
            model_outputs = []
            loss_values = []
            model_gradients = []
            #temp_loss_values = []
            # Load data to device
            member_input, member_target = member_input.float().to(self.device) \
                , member_target.long().to(self.device)


            data, labels =member_input, member_target

            # Create one-hot encoding of labels, as this has to be done only once.
            one_hot_labels = encode_labels(labels, self.one_hot_encoding, self.device).float()


            # For each observed model collect input data for the attack model.
            for model in target_models:

                model.eval()
                model.to(self.device)
                optimizer = optimizers.__dict__[self.optimizer_name](
                    params=model.parameters(),
                    lr=self.learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay
                )

                # Get predictions
                predictions = model(data)
                temp_pred = torch.from_numpy(predictions.data.cpu().numpy()).to(self.device)
                model_outputs.append(temp_pred.requires_grad_(True))  # ba cheng yuan he fei chen yuan
                # calculate loss values
                # predictions 64*10 one_hot_labels 64*10
                loss_value = torch.sum(predictions * one_hot_labels, dim=1).view([-1, 1])
                ###loss_value是對應標籤的模型輸出值
                temp_loss = torch.from_numpy(loss_value.data.cpu().numpy()).to(self.device)

                loss_values.append(temp_loss.requires_grad_(True))

                gradients = torch.zeros(0)
                # dui mei yige yangebn qiu loss he tidu
                # qian 32 ge shi cheng yaun hou 32 ge fei cheng yuan
                for index in range(predictions.size(0)):
                    # ji suan yi ge shi ti de loss, qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                    loss = self.loss_function(predictions[index].view([1, -1]), labels[index].view([-1]))

                    optimizer.zero_grad()
                    if index == (predictions.size(0)) - 1:
                        loss.backward(retain_graph=False)
                    else:
                        loss.backward(retain_graph=True)
                    # current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])#qu zui hou yi ceng can shu de tidu
                    if self.model_name == "AlexNet":
                        current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name == "CNNCifar10":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    elif self.model_name == "CNNFashion_Mnist":
                        current_grads = model.fc.weight.grad.view([1, 1, 7 * 7 * 32, self.number_of_classes])
                    elif self.model_name == "CNNMnist":
                        current_grads = model.fc2.weight.grad.view([1, 1, 50, self.number_of_classes])
                    elif self.model_name == "LeNet5":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    # current_grads = model.lastlayer_name.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name=="CNNModel":
                        current_grads = model.fc2.weight.grad.view([1, 1, 256, self.number_of_classes])
                    else:
                        current_grads = None
                    if gradients.size()[0] == 0:
                        gradients = current_grads
                    else:
                        gradients = torch.cat((gradients, current_grads))

                temp_gradients = torch.from_numpy(gradients.data.cpu().numpy()).to(self.device)
                model_gradients.append(temp_gradients.requires_grad_(True))
                model.to('cpu')



            self.attack_model.to(self.device)
            self.attack_optimizer.zero_grad()


            features_in_hook.clear()
            features_out_hook.clear()

            output=self.attack_model(model_outputs, one_hot_labels, loss_values, model_gradients)
            #encoder_outputs.append(features_out_hook[-2].squeeze().detach().cpu().numpy())
            encoder_outputs.append(features_out_hook[-1].squeeze().detach().cpu().numpy())
            features_in_hook.clear()
            features_out_hook.clear()

            self.attack_model.to('cpu')

        for member_input, member_target in tqdm(temp_data_loader_2):
            times += 1

            batch_confusion_matrix.reset()

            # Pre-define variables to use.
            model_outputs = []
            loss_values = []
            model_gradients = []
            # temp_loss_values = []
            # Load data to device
            member_input, member_target = member_input.float().to(self.device) \
                , member_target.long().to(self.device)

            data, labels = member_input, member_target

            # Create one-hot encoding of labels, as this has to be done only once.
            one_hot_labels = encode_labels(labels, self.one_hot_encoding, self.device).float()

            # For each observed model collect input data for the attack model.
            for model in target_models:

                model.eval()
                model.to(self.device)
                optimizer = optimizers.__dict__[self.optimizer_name](
                    params=model.parameters(),
                    lr=self.learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay
                )

                # Get predictions
                predictions = model(data)
                temp_pred = torch.from_numpy(predictions.data.cpu().numpy()).to(self.device)
                model_outputs.append(temp_pred.requires_grad_(True))  # ba cheng yuan he fei chen yuan
                # calculate loss values
                # predictions 64*10 one_hot_labels 64*10
                loss_value = torch.sum(predictions * one_hot_labels, dim=1).view([-1, 1])
                ###loss_value是對應標籤的模型輸出值
                temp_loss = torch.from_numpy(loss_value.data.cpu().numpy()).to(self.device)

                loss_values.append(temp_loss.requires_grad_(True))

                gradients = torch.zeros(0)
                # dui mei yige yangebn qiu loss he tidu
                # qian 32 ge shi cheng yaun hou 32 ge fei cheng yuan
                for index in range(predictions.size(0)):
                    # ji suan yi ge shi ti de loss, qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                    loss = self.loss_function(predictions[index].view([1, -1]), labels[index].view([-1]))

                    optimizer.zero_grad()
                    if index == (predictions.size(0)) - 1:
                        loss.backward(retain_graph=False)
                    else:
                        loss.backward(retain_graph=True)
                    # current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])#qu zui hou yi ceng can shu de tidu
                    if self.model_name == "AlexNet":
                        current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name == "CNNCifar10":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    elif self.model_name == "CNNFashion_Mnist":
                        current_grads = model.fc.weight.grad.view([1, 1, 7 * 7 * 32, self.number_of_classes])
                    elif self.model_name == "CNNMnist":
                        current_grads = model.fc2.weight.grad.view([1, 1, 50, self.number_of_classes])
                    elif self.model_name == "LeNet5":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    # current_grads = model.lastlayer_name.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name == "CNNModel":
                        current_grads = model.fc2.weight.grad.view([1, 1, 256, self.number_of_classes])
                    else:
                        current_grads = None
                    if gradients.size()[0] == 0:
                        gradients = current_grads
                    else:
                        gradients = torch.cat((gradients, current_grads))

                temp_gradients = torch.from_numpy(gradients.data.cpu().numpy()).to(self.device)
                model_gradients.append(temp_gradients.requires_grad_(True))
                model.to('cpu')

            self.attack_model.to(self.device)
            self.attack_optimizer.zero_grad()

            features_in_hook.clear()
            features_out_hook.clear()

            output = self.attack_model(model_outputs, one_hot_labels, loss_values, model_gradients)
            #encoder_outputs.append(features_out_hook[-2].squeeze().detach().cpu().numpy())
            encoder_outputs.append(features_out_hook[-1].squeeze().detach().cpu().numpy())
            features_in_hook.clear()
            features_out_hook.clear()

            # Place model back to CPU
            self.attack_model.to('cpu')

        plt.figure()
        tsne = TSNE(n_components=2, n_iter=20000, perplexity=30)
        features_tsne = tsne.fit_transform(encoder_outputs)
        co = []
        for i in range(len(member)):
            co.append('b')
        for i in range(len(non_member)):
            co.append('r')
        # 绘制 t-SNE 可视化图
        plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=co, cmap='viridis')
        model_path_split = model_path.split('/')
        title='t-SNE Visualization'+model_path_split[-2]
        plt.title(title)
        class1_patch = plt.Line2D([0], [0], marker='o', color='w', label='member', markerfacecolor='blue',
                                  markersize=10)
        class2_patch = plt.Line2D([0], [0], marker='o', color='w', label='nonmember', markerfacecolor='red',
                                  markersize=10)
        plt.legend(handles=[class1_patch, class2_patch])
        #plt.show()

        with open('./log/persistent.txt', 'r', encoding="utf-8") as f:
            read_data = int(f.read())
            file_number = str(read_data + 1)


        name = 'tsne-' + model_path_split[-2]

        name = self.save_model_patch + name + file_number + '.png'

        plt.savefig(name)







    def get_member_nomember_infomation(self):
        member, non_member = self.attack_test_data
        member_data_loader = DataLoader(member,
                                      batch_size=self.attack_train_batch_size // 2,
                                      shuffle=True,
                                      num_workers=0,
                                      pin_memory=True
                                      )

        nonmember_data_loader = DataLoader(non_member,
                                        batch_size=self.attack_train_batch_size // 2,
                                        shuffle=True,
                                        num_workers=0,
                                        pin_memory=True
                                        )

        l2norm_member_bacth=[]
        l2norm_nonmember_bacth=[]
        loss_member_batch=[]
        loss_nonmember_batch=[]
        self.model.to(self.device)
        self.model.eval()
        for member_input, member_target in member_data_loader:
            self.optimizer.zero_grad()
            member_input, member_target = member_input.float().to(self.device), member_target.long().to(self.device)

            outputs = self.model(member_input)
            loss = self.loss_function(outputs, member_target)

            loss_member_batch.append(loss.item())
            loss.backward()
            if self.model_name == "AlexNet":
                current_grads = self.model.classifier.weight.grad
            elif self.model_name == "CNNCifar10":
                current_grads = self.model.fc3.weight.grad
            elif self.model_name == "CNNFashion_Mnist":
                current_grads = self.model.fc.weight.grad
            elif self.model_name == "CNNMnist":
                current_grads = self.model.fc2.weight.grad
            elif self.model_name == "LeNet5":
                current_grads = self.model.fc3.weight.grad
            elif self.model_name == "CNNModel":
                current_grads = self.model.fc2.weight.grad

            # current_grads = model.lastlayer_name.weight.grad.view([1, 1, 256, self.number_of_classes])
            else:
                current_grads = None
            l2_norm_member = torch.norm(current_grads, p=2)
            l2norm_member_bacth.append(l2_norm_member.item())

        for nonmember_input, nonmember_target in nonmember_data_loader:
            self.optimizer.zero_grad()
            nonmember_input, nonmember_target = nonmember_input.float().to(self.device), nonmember_target.long().to(self.device)

            outputs = self.model(nonmember_input)
            loss = self.loss_function(outputs, nonmember_target)

            loss_nonmember_batch.append(loss.item())
            loss.backward()
            if self.model_name == "AlexNet":
                current_grads = self.model.classifier.weight.grad
            elif self.model_name == "CNNCifar10":
                current_grads = self.model.fc3.weight.grad
            elif self.model_name == "CNNFashion_Mnist":
                current_grads = self.model.fc.weight.grad
            elif self.model_name == "CNNMnist":
                current_grads = self.model.fc2.weight.grad
            elif self.model_name == "LeNet5":
                current_grads = self.model.fc3.weight.grad
            elif self.model_name == "CNNModel":
                current_grads = self.model.fc2.weight.grad
            # current_grads = model.lastlayer_name.weight.grad.view([1, 1, 256, self.number_of_classes])
            else:
                current_grads = None
            l2_norm_nonmember = torch.norm(current_grads, p=2)

            l2norm_nonmember_bacth.append(l2_norm_nonmember.item())

        self.model.to('cpu')

        l2_norm_member_round_avg=sum(l2norm_member_bacth)/len(l2norm_member_bacth)
        l2_norm_nonmember_round_avg=sum(l2norm_nonmember_bacth)/len(l2norm_nonmember_bacth)
        loss_member_round_avg=sum(loss_member_batch)/len(loss_member_batch)
        loss_nonmember_round_avg=sum(loss_nonmember_batch)/len(loss_nonmember_batch)
        self.l2norm_member_round.append(l2_norm_member_round_avg)
        self.l2norm_nonmember_round.append(l2_norm_nonmember_round_avg)
        self.loss_member_round.append(loss_member_round_avg)
        self.loss_nonmember_round.append(loss_nonmember_round_avg)
        log_l2norm_member = logging.getLogger('log_l2norm_member')
        log_l2norm_nonmember = logging.getLogger('log_l2norm_nonmember')
        log_loss_member = logging.getLogger('log_loss_member')
        log_loss_nomember = logging.getLogger('log_loss_nomember')
        log_l2norm_member.info(l2_norm_member_round_avg)
        log_l2norm_nonmember.info(l2_norm_nonmember_round_avg)
        log_loss_member.info(loss_member_round_avg)
        log_loss_nomember.info(loss_nonmember_round_avg)


    def plot_img(self,target_models):
        member, non_member = self.attack_test_data
        member_data_loader = DataLoader(member,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=0,
                                      pin_memory=True
                                      )

        nonmember_data_loader = DataLoader(non_member,
                                        batch_size=1,
                                        shuffle=True,
                                        num_workers=0,
                                        pin_memory=True
                                        )
        count=0
        for model in target_models:
            count+=1
            l2norm_member_bacth=[]
            l2norm_nonmember_bacth=[]
            loss_member_batch=[]
            loss_nonmember_batch=[]
            model=model.to(self.device)
            model.eval()
            for member_input, member_target in member_data_loader:
                model.zero_grad()
                member_input, member_target = member_input.float().to(self.device), member_target.long().to(self.device)

                outputs = model(member_input)
                loss = self.loss_function(outputs, member_target)

                loss_member_batch.append(loss.item())
                loss.backward()
                if self.model_name == "AlexNet":
                    current_grads = model.classifier.weight.grad
                elif self.model_name == "CNNCifar10":
                    current_grads = model.fc3.weight.grad
                elif self.model_name == "CNNFashion_Mnist":
                    current_grads = model.fc.weight.grad
                elif self.model_name == "CNNMnist":
                    current_grads = model.fc2.weight.grad
                elif self.model_name == "LeNet5":
                    current_grads = model.fc3.weight.grad
                elif self.model_name == "CNNModel":
                    current_grads = model.fc2.weight.grad
                # current_grads = model.lastlayer_name.weight.grad.view([1, 1, 256, self.number_of_classes])
                else:
                    current_grads = None
                l2_norm_member = torch.norm(current_grads, p=2)
                l2norm_member_bacth.append(l2_norm_member.item())

            for nonmember_input, nonmember_target in nonmember_data_loader:
                model.zero_grad()
                nonmember_input, nonmember_target = nonmember_input.float().to(self.device), nonmember_target.long().to(self.device)

                outputs = model(nonmember_input)
                loss = self.loss_function(outputs, nonmember_target)

                loss_nonmember_batch.append(loss.item())
                loss.backward()
                if self.model_name == "AlexNet":
                    current_grads = model.classifier.weight.grad
                elif self.model_name == "CNNCifar10":
                    current_grads = model.fc3.weight.grad
                elif self.model_name == "CNNFashion_Mnist":
                    current_grads = model.fc.weight.grad
                elif self.model_name == "CNNMnist":
                    current_grads = model.fc2.weight.grad
                elif self.model_name == "LeNet5":
                    current_grads = model.fc3.weight.grad
                elif self.model_name == "CNNModel":
                    current_grads = model.fc2.weight.grad
                # current_grads = model.lastlayer_name.weight.grad.view([1, 1, 256, self.number_of_classes])
                else:
                    current_grads = None
                l2_norm_nonmember = torch.norm(current_grads, p=2)

                l2norm_nonmember_bacth.append(l2_norm_nonmember.item())

            model.to('cpu')
            plt.figure()
            l2_min = 0
            l2_max = int(max(l2norm_nonmember_bacth))
            bins = int(l2_max / 0.5)
            counts_p,_,_=plt.hist(x=l2norm_member_bacth,bins=bins,range=(l2_min,l2_max),density=True)
            counts_q, _, _ = plt.hist(x=l2norm_nonmember_bacth, bins=bins, range=(l2_min, l2_max), density=True)


            plt.figure()
            total_p = sum(counts_p)
            total_q = sum(counts_q)
            p = [count / total_p for count in counts_p]
            q = [count / total_q for count in counts_q]
            js_distance = distance.jensenshannon(p, q)
            js_distance = round(js_distance, 5)
            #print(js_distance)
            #print(l2norm_member_bacth)
            #print(l2norm_nonmember_bacth)
            #plt.hist(x=(l2norm_member_bacth,l2norm_nonmember_bacth),label=['l2norm_member','l2norm_nonmember'],bins=bins,range=(0,l2_max),density=False,color=['b','r'],stacked=False)
            #plt.hist(x=l2norm_member_bacth,label='l2norm_member',density=False,color='b')
            plt.hist(x=(l2norm_member_bacth, l2norm_nonmember_bacth), label=['l2norm_member', 'l2norm_nonmember'],
                     width=0.3, bins=bins, range=(0, l2_max), density=False, color=['b', 'r'], stacked=False)
            plt.legend()
            plt.xticks(np.arange(0, l2_max, 5))
            #plt.yticks(np.arange(0, 800, 50))
            #plt.ylim(0, 100)
            #plt.hist(x=l2norm_member_bacth,label='l2norm_member',bins=bins,range=(l2_min,l2_max),density=True,color='r',histtype='barstacked')
            #plt.hist(x=l2norm_nonmember_bacth,label='l2norm_nonmember',bins=bins,range=(l2_min,l2_max),density=True,color='b',histtype='barstacked')
            '''
            #L2normDistribution-Cifar100-AlexNet-BackdoorTrainPatch
            #LossDistribution-Cifar100-AlexNet-BackdoorTrainPatch
            #LossDistribution-Cifar10-AlexNet-GradientAscent
            #L2normDistribution-Cifar100-AlexNet-GradientAscent
            #gradient_ascent
            loss_min=0;
            loss_max=int(max(loss_nonmember_batch))
            loss_max=13
            if loss_max>300:
                loss_max=120
            bins=int(loss_max/0.5)
            plt.figure()
            plt.hist(x=loss_member_batch, label='loss_member',
                     bins=bins, range=(0, loss_max), density=False, color='b',alpha=0.3)
            plt.hist(x=loss_nonmember_batch, label='loss_nonmember',
                     bins=bins, range=(0, loss_max), density=False, color='r',alpha=0.3)
            plt.legend()
            plt.xlabel('value')
            plt.xticks(np.arange(0, loss_max, 0.5))
            plt.ylabel('frequency')
            
            
            l2_min = 0
            print(max(l2norm_nonmember_bacth))
            #l2_max = int(max(l2norm_nonmember_bacth))
            l2_max = 100
            bins = int(l2_max / 0.5)
            plt.figure()
            plt.hist(l2norm_member_bacth, label='l2norm_member',bins=bins, range=(0, l2_max), density=False, color='b', alpha=0.3)
            plt.hist(l2norm_nonmember_bacth, label='l2norm_nonmember',bins=bins, range=(0, l2_max), density=False, color='r', alpha=0.3)
            plt.legend()
            plt.xticks(np.arange(0, l2_max, 2))
            plt.xlabel('value')
            plt.ylabel('frequency')
            
            '''
            plt.xlabel('value')
            plt.ylabel('frequency')

            with open('./log/persistent.txt', 'r', encoding="utf-8") as f:
                read_data = int(f.read())
                file_number = str(read_data + 1)
            name = str(js_distance)+'-'+'l2norm-' +str(count)+'-'+file_number
            plt.title(name)
            name = self.save_model_patch + name  + '.png'
            plt.legend()  # 添加图例
            plt.savefig(name)


            plt.figure()
            loss_min=0;
            loss_max=int(max(loss_nonmember_batch))
            bins=int(loss_max/0.5)
            counts_p, _, _ = plt.hist(x=loss_member_batch, bins=bins, range=(loss_min, loss_max), density=False)
            counts_q, _, _ = plt.hist(x=loss_nonmember_batch, bins=bins, range=(loss_min, loss_max), density=False)

            plt.figure()
            total_p = sum(counts_p)
            total_q = sum(counts_q)
            p = [count / total_p for count in counts_p]
            q = [count / total_q for count in counts_q]
            js_distance = distance.jensenshannon(p, q)
            js_distance = round(js_distance, 5)
            #print(js_distance)
            #print(loss_member_batch)
            #print(loss_nonmember_batch)
            plt.hist(x=(loss_member_batch,loss_nonmember_batch),label=['loss_member','loss_nonmember',str(js_distance)],bins=bins,range=(0,loss_max),density=False,color=['b','r'],stacked=False)
            #plt.hist(x=loss_member_batch,
            #         label='loss_member',
            #         density=False, color='b')

            #plt.ylim(0,100)
            plt.xlabel('value')
            plt.ylabel('density')

            with open('./log/persistent.txt', 'r', encoding="utf-8") as f:
                read_data = int(f.read())
                file_number = str(read_data + 1)
            name = str(js_distance)+'-'+'loss-' +str(count)+'-'+file_number
            plt.title(name)
            name = self.save_model_patch + name  + '.png'
            plt.legend()  # 添加图例
            plt.savefig(name)


    def plot_backdoor_img(self):
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
                    image3[:, y, x] = p1 * image1[:, y, x] + p2 * image2[:, y, x]
            # print("image3",image3.shape) #torch.Size([1, 28, 28])
            # scipy.misc.imsave(name3, image3)
            return image3

        def filter2(fname1, fname2):
            p1 = float(0)
            p2 = 1 - p1

            image3 = weighted_part_average(fname1, fname2, fname1, p1, p2)

            return image3

        fig, axes = plt.subplots(10, 5)  # 1行5列的子图，适当调整图形大小
        # fig, axes = plt.subplots(10, 5)  # 1行5列的子图，适当调整图形大小
        member, non_member = self.attack_test_data
        pltclasses = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        for i in range(10):
            labal = member[i * 200 + 5][1]
            three_index = len(self.attack_patch_list) // 3
            image = member[i * 200 + 5][0]

            image = np.transpose(image, (1, 2, 0))

            axes[i][0].imshow(image)
            axes[i][0].set_title(pltclasses[labal])
            axes[i][0].axis('off')
            # plt.show()

            # 循环绘制多个子图
            # for j in range(4):
            image = member[i * 200 + 5][0]
            applied_patch = self.attack_patch_list[0]
            applied_patch = torch.from_numpy(applied_patch)
            p_img_batch = filter2(image, applied_patch)
            # image = torch.from_numpy(perturbated_image)
            image = p_img_batch.numpy()
            image = np.transpose(image, (1, 2, 0))
            axes[i][1].imshow(image)
            axes[i][1].set_title(pltclasses[labal])
            axes[i][1].axis('off')

            image = member[i * 200 + 5][0]
            applied_patch = self.attack_patch_list[three_index * 1]
            applied_patch = torch.from_numpy(applied_patch)

            p_img_batch = filter2(image, applied_patch)

            # image = torch.from_numpy(perturbated_image)
            image = p_img_batch.numpy()
            image = np.transpose(image, (1, 2, 0))
            axes[i][2].imshow(image)
            axes[i][2].set_title(pltclasses[labal])
            axes[i][2].axis('off')

            image = member[i * 200 + 5][0]
            applied_patch = self.attack_patch_list[three_index * 2]
            applied_patch = torch.from_numpy(applied_patch)

            p_img_batch = filter2(image, applied_patch)

            # image = torch.from_numpy(perturbated_image)
            image = p_img_batch.numpy()
            image = np.transpose(image, (1, 2, 0))
            axes[i][3].imshow(image)
            axes[i][3].set_title(pltclasses[labal])
            axes[i][3].axis('off')

            image = member[i * 200 + 5][0]
            applied_patch = self.attack_patch_list[-1]
            applied_patch = torch.from_numpy(applied_patch)

            p_img_batch = filter2(image, applied_patch)

            # image = torch.from_numpy(perturbated_image)
            image = p_img_batch.numpy()
            image = np.transpose(image, (1, 2, 0))
            axes[i][4].imshow(image)
            axes[i][4].set_title(pltclasses[labal])
            axes[i][4].axis('off')
        plt.subplots_adjust(hspace=0.5)
        plt.subplots_adjust(wspace=0.1)
        plt.subplot_tool()
        plt.show()



    


    def test_attack(self, round_number, target_models):
        """ Ja hier moet dus documentatie """
        # Metrics
        start_time = time.time()
        final_confusion_matrix = ConfusionMatrix()
        batch_confusion_matrix = ConfusionMatrix()
        losses = AverageMeter()
        accuracy = AverageMeter()
        # create dict for keeping track of scores per class.
        class_matrices = dict((data_class, ConfusionMatrix()) for data_class in range(self.number_of_classes))

        self.attack_model.eval()

        member, non_member = self.attack_test_data
        temp_data_loader = DataLoader(member,
                                      batch_size=self.attack_test_batch_size // 2,
                                      shuffle=False,
                                      num_workers=2,
                                      pin_memory=True
                                      )

        temp_data_loader_2 = DataLoader(non_member,
                                        batch_size=self.attack_test_batch_size // 2,
                                        shuffle=False,
                                        num_workers=2,
                                        pin_memory=True
                                        )

        data_loader = zip(temp_data_loader, temp_data_loader_2)
        times=0
        for (member_input, member_target), (non_member_input, non_member_target) in data_loader:
            times+=1
            #
            batch_start_time = time.time()
            batch_confusion_matrix.reset()
            #
            model_outputs = []
            loss_values = []
            model_gradients = []

            # Load data to device
            member_input, member_target = member_input.float().to(self.device) \
                , member_target.long().to(self.device)
            non_member_input, non_member_target = non_member_input.float().to(self.device) \
                , non_member_target.long().to(self.device)

            data, labels = torch.cat((member_input, non_member_input)), torch.cat(
                (member_target, non_member_target))

            # Create one-hot encoding of labels, as this has to be done only once.
            one_hot_labels = encode_labels(labels, self.one_hot_encoding, self.device).float()
            #
            for model in target_models:
                #
                model.eval()
                model.to(self.device)
                optimizer = optimizers.__dict__[self.optimizer_name](
                    params=model.parameters(),
                    lr=self.learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay
                )

                # Get predictions
                predictions = model(data)
                temp_pred = torch.from_numpy(predictions.data.cpu().numpy()).to(self.device)
                model_outputs.append(temp_pred.requires_grad_(True))
                # calculate loss values
                loss_value = torch.sum(predictions * one_hot_labels, dim=1).view([-1, 1])
                temp_loss = torch.from_numpy(loss_value.data.cpu().numpy()).to(self.device)
                loss_values.append(temp_loss.requires_grad_(True))

                gradients = torch.zeros(0)
                for index in range(predictions.size(0)):
                    loss = self.loss_function(predictions[index].view([1, -1]), labels[index].view([-1]))
                    optimizer.zero_grad()
                    if index == (predictions.size(0)) - 1:
                        loss.backward(retain_graph=False)
                    else:
                        loss.backward(retain_graph=True)
                    # current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])
                    if self.model_name == "AlexNet":
                        current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name == "CNNCifar10":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    elif self.model_name == "CNNFashion_Mnist":
                        current_grads = model.fc.weight.grad.view([1, 1, 7 * 7 * 32, self.number_of_classes])
                    elif self.model_name == "CNNMnist":
                        current_grads = model.fc2.weight.grad.view([1, 1, 50, self.number_of_classes])
                    elif self.model_name == "LeNet5":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    elif self.model_name == "CNNModel":
                        current_grads = model.fc2.weight.grad.view([1, 1, 256, self.number_of_classes])
                    # current_grads = model.lastlayer_name.weight.grad.view([1, 1, 256, self.number_of_classes])
                    else:
                        current_grads = None
                    if gradients.size()[0] == 0:
                        gradients = current_grads
                    else:
                        gradients = torch.cat((gradients, current_grads))

                temp_gradients = torch.from_numpy(gradients.data.cpu().numpy()).to(self.device)
                model_gradients.append(temp_gradients.requires_grad_(True))
                model.to('cpu')

            # Get the predictions for the membership classification
            self.attack_model.to(self.device)
            with torch.no_grad():
                # Get membership predictions
                membership_predictions = self.attack_model(model_outputs, one_hot_labels, loss_values,
                                                           model_gradients)

                # Change labels of the data for binary attack classification
                member_target = torch.Tensor([1 for _ in member_target])
                non_member_target = torch.Tensor([0 for _ in non_member_target])
                membership_labels = torch.cat((member_target, non_member_target))
                membership_labels = torch.unsqueeze(membership_labels, -1).data.float().to(self.device)

                # Calculate the loss of attack model
                attack_loss = self.attack_loss_function(membership_predictions, membership_labels)

                # update class confusion matrices, to see scores per original class
                combined_labels = zip(labels, membership_labels)
                for index, (data_class, member_class) in enumerate(combined_labels):
                    class_matrices[data_class.item()].update(membership_predictions[index], member_class)

                batch_confusion_matrix.update(membership_predictions, membership_labels)
                accuracy.update(batch_confusion_matrix.get_accuracy(), self.attack_test_batch_size)
                losses.update(attack_loss.item(), self.attack_test_batch_size)

            self.attack_model.to('cpu')

            temp = batch_confusion_matrix.get_confusion_matrix()
            final_confusion_matrix.update_from_matrix(temp)
            message = f'[ Round: {round_number} ' \
                      f'| Attacker Test ' \
                      f'| Batch Time: {time.time() - batch_start_time:.2f}s ' \
                      f'| Batch Loss: {attack_loss:.5f} ' \
                      f'| Batch Acc: {batch_confusion_matrix.get_accuracy():.2f}% ]' \
                      f'| Class Conf Matrix: TP:{temp[0]}' \
                      f' FP:{temp[1]}' \
                      f' TN:{temp[2]}' \
                      f' FN:{temp[3]} ]'
            logging.info(message)
            if times%self.save_information_count==0:
                log_attack = logging.getLogger('log_attack')
                log_attack.info(message)

        # Print the final summary of this test
        temp = final_confusion_matrix.get_confusion_matrix()
        self.best_inference_acc_test=max(self.best_inference_acc_test,final_confusion_matrix.get_accuracy())
        message = f'[ Round: {round_number} Totals ' \
                  f'| Attacker Test ' \
                  f'| Total Time: {time.time() - start_time:.2f}s ' \
                  f'| Loss: {losses.avg:.5f} ' \
                  f'| Acc: {final_confusion_matrix.get_accuracy():.2f}% ' \
                  f'| Best Test Acc: {self.best_inference_acc_test:.2f}% ' \
                  f'| Best Train Acc: {self.best_inference_acc_train:.2f}% ' \
                  f'| Final Conf Matrix: TP:{temp[0]}' \
                  f' FP:{temp[1]}' \
                  f' TN:{temp[2]}' \
                  f' FN:{temp[3]} ]'
        logging.info(message)
        log_attack = logging.getLogger('log_attack')
        log_attack.info(message)

        # Print all the scores for each class.
        message = f'[ Round: {round_number} Class Scores ]\n'
        for data_class in range(self.number_of_classes):
            class_matrix = class_matrices.get(data_class)
            scores = class_matrix.get_confusion_matrix()
            class_message = f'[ Class {data_class}' \
                            f'| Class Acc: {class_matrix.get_accuracy():.2f}% ' \
                            f'| Conf Matrix: TP:{scores[0]}' \
                            f' FP:{scores[1]}' \
                            f' TN:{scores[2]}' \
                            f' FN:{scores[3]} ]'
            if data_class == 99:
                message += class_message
            else:
                message += class_message + '\n'
        logging.info(message)
        log_attack = logging.getLogger('log_attack')
        log_attack.info(message)

        return losses.avg, final_confusion_matrix.get_accuracy()

    def test_attack1(self, round_number, target_models):
            """ Ja hier moet dus documentatie """
            # Metrics
            start_time = time.time()
            final_confusion_matrix = ConfusionMatrix()
            batch_confusion_matrix = ConfusionMatrix()
            losses = AverageMeter()
            accuracy = AverageMeter()
            # create dict for keeping track of scores per class.
            class_matrices = dict((data_class, ConfusionMatrix()) for data_class in range(self.number_of_classes))

            self.attack_model.eval()

            member, non_member = self.attack_test_data
            temp_data_loader = DataLoader(member,
                                        batch_size=self.attack_test_batch_size // 2,
                                        shuffle=False,
                                        num_workers=2,
                                        pin_memory=True
                                        )

            temp_data_loader_2 = DataLoader(non_member,
                                            batch_size=self.attack_test_batch_size // 2,
                                            shuffle=False,
                                            num_workers=2,
                                            pin_memory=True
                                            )
            
            member_poison=self.attack_test_member_poison
            non_member_poison=self.attack_test_non_member_poison

            temp_data_loader_poison=DataLoader(member_poison,
                                            batch_size=self.attack_train_batch_size // 2,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True
            )


            temp_data_loader_2_poison=DataLoader(non_member_poison,
                                            batch_size=self.attack_train_batch_size // 2,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True
            )

            data_loader = zip(temp_data_loader, temp_data_loader_2,temp_data_loader_poison,temp_data_loader_2_poison)
            #data_loader_poison=zip(temp_data_loader_poison,temp_data_loader_2_poison)
            ####################################################
            # bao cun mei ge batch de ti du fan shu he loss
            l2_norm_member_list = []
            l2_norm_nonmember_list = []
            loss_member_list = []
            loss_nonmember_list = []



            l2_norm_member_poison_list = []
            l2_norm_nonmember_poison_list = []
            loss_member_poison_list = []
            loss_nonmember_poison_list = []


            temp = self.batch_size
            halftbatch = temp // 2  # xu yao geng ju batch zhi tiaozheng!!!
            times = 0
            ####################################################
            for (member_input, member_target), (non_member_input, non_member_target),(member_input_poison, member_target_poison),(non_member_input_poison, non_member_target_poison) in data_loader:
                times+=1
                #
                batch_start_time = time.time()
                batch_confusion_matrix.reset()
                segmation = len(member_input)
                # Pre-define variables to use.
                model_outputs = []
                loss_values = []
                model_gradients = []



                model_outputs_poison = []
                loss_values_poison = []
                model_gradients_poison = []

                #temp_loss_values = []
                # Load data to device
                member_input, member_target = member_input.float().to(self.device) \
                    , member_target.long().to(self.device)
                non_member_input, non_member_target = non_member_input.float().to(self.device) \
                    , non_member_target.long().to(self.device)
                
                member_input_poison, member_target_poison = member_input_poison.float().to(self.device) \
                    , member_target_poison.long().to(self.device)
                non_member_input_poison, non_member_target_poison = non_member_input_poison.float().to(self.device) \
                    , non_member_target_poison.long().to(self.device)

                data, labels = torch.cat((member_input, non_member_input)), torch.cat(
                    (member_target, non_member_target))

                # Create one-hot encoding of labels, as this has to be done only once.
                one_hot_labels = encode_labels(labels, self.one_hot_encoding, self.device).float()

                data_poison, labels_poison = torch.cat((member_input_poison, non_member_input_poison)), torch.cat(
                    (member_target_poison, non_member_target_poison))
                #plt.imshow(data_poison[0].cpu().numpy().transpose(1,2,0))
                #plt.show()
                one_hot_labels_poison = encode_labels(labels_poison, self.one_hot_encoding, self.device).float()
                ####################
                loss_member_eachbatch=[]
                loss_nomember_eachbatch=[]

                loss_member_eachbatch_poison=[]
                loss_nomember_eachbatch_poison=[]
                ####################
                # For each observed model collect input data for the attack model.

                loss_labal_ori_list=[]

                for model in target_models:
                    #
                    ####################################################
                    # qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                    member_loss = []
                    nonmember_loss = []
                    ####################################################
                    model.eval()
                    model.to(self.device)
                    optimizer = optimizers.__dict__[self.optimizer_name](
                        params=model.parameters(),
                        lr=self.learning_rate,
                        momentum=self.momentum,
                        weight_decay=self.weight_decay
                    )

                    # Get predictions
                    predictions = model(data)
                    temp_pred = torch.from_numpy(predictions.data.cpu().numpy()).to(self.device)
                    model_outputs.append(temp_pred.requires_grad_(True))  # ba cheng yuan he fei chen yuan
                    # calculate loss values
                    # predictions 64*10 one_hot_labels 64*10
                    loss_value = torch.sum(predictions * one_hot_labels, dim=1).view([-1, 1])
                    ###loss_value是對應標籤的模型輸出值
                    temp_loss = torch.from_numpy(loss_value.data.cpu().numpy()).to(self.device)

                    loss_values.append(temp_loss.requires_grad_(True))
                    loss_labal=[]
                    gradients = torch.zeros(0)
                    # dui mei yige yangebn qiu loss he tidu
                    # qian 32 ge shi cheng yaun hou 32 ge fei cheng yuan
                    for index in range(predictions.size(0)):
                        # ji suan yi ge shi ti de loss, qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                        loss = self.loss_function(predictions[index].view([1, -1]), labels[index].view([-1]))
                        loss_labal.append(loss)
                        ################## add loss into list
                        if index < segmation:
                            member_loss.append(loss)
                        else:
                            nonmember_loss.append(loss)
                        ##################
                        optimizer.zero_grad()
                        if index == (predictions.size(0)) - 1:
                            loss.backward(retain_graph=False)
                        else:
                            loss.backward(retain_graph=True)
                        # current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])#qu zui hou yi ceng can shu de tidu
                        if self.model_name == "AlexNet":
                            current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])
                        elif self.model_name == "CNNCifar10":
                            current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                        elif self.model_name == "CNNFashion_Mnist":
                            current_grads = model.fc.weight.grad.view([1, 1, 7 * 7 * 32, self.number_of_classes])
                        elif self.model_name == "CNNMnist":
                            current_grads = model.fc2.weight.grad.view([1, 1, 50, self.number_of_classes])
                        elif self.model_name == "LeNet5":
                            current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                        # current_grads = model.lastlayer_name.weight.grad.view([1, 1, 256, self.number_of_classes])
                        elif self.model_name=="CNNModel":
                            current_grads = model.fc2.weight.grad.view([1, 1, 256, self.number_of_classes])
                        else:
                            current_grads = None
                        if gradients.size()[0] == 0:
                            gradients = current_grads
                        else:
                            gradients = torch.cat((gradients, current_grads))
                    # 64, 1, 256, 100 qian 32 ge shi chengyuan zuihouyicengde tidu hou 32shi fei cheng yuan de tidu
                    temp_gradients = torch.from_numpy(gradients.data.cpu().numpy()).to(self.device)
                    loss_labal_ori_list.append(torch.unsqueeze(torch.tensor(loss_labal), dim=1))

                    model_gradients.append(temp_gradients.requires_grad_(True))
                    loss_member_eachbatch.append(member_loss)
                    loss_nomember_eachbatch.append(nonmember_loss)
                    model.to('cpu')
                



                loss_labal_poison_list=[]
                for model in target_models:
                    #
                    ####################################################
                    # qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                    member_loss_poison = []
                    nonmember_loss_poison = []
                    ####################################################
                    model.eval()
                    model.to(self.device)
                    optimizer = optimizers.__dict__[self.optimizer_name](
                        params=model.parameters(),
                        lr=self.learning_rate,
                        momentum=self.momentum,
                        weight_decay=self.weight_decay
                    )

                    # Get predictions
                    predictions = model(data_poison)
                    _, predicted = torch.max(predictions, 1)

                    temp_pred = torch.from_numpy(predictions.data.cpu().numpy()).to(self.device)
                    model_outputs_poison.append(temp_pred.requires_grad_(True))  # ba cheng yuan he fei chen yuan
                    # calculate loss values
                    # predictions 64*10 one_hot_labels 64*10
                    loss_value = torch.sum(predictions * one_hot_labels_poison, dim=1).view([-1, 1])

                    ###loss_value是對應標籤的模型輸出值
                    temp_loss = torch.from_numpy(loss_value.data.cpu().numpy()).to(self.device)

                    loss_values_poison.append(temp_loss.requires_grad_(True))

                    gradients = torch.zeros(0)
                    # dui mei yige yangebn qiu loss he tidu
                    # qian 32 ge shi cheng yaun hou 32 ge fei cheng yuan
                    loss_labal=[]
                    for index in range(predictions.size(0)):
                        # ji suan yi ge shi ti de loss, qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                        loss = self.loss_function(predictions[index].view([1, -1]), labels[index].view([-1]))
                        loss_labal.append(loss)
                        ################## add loss into list
                        if index < segmation:
                            member_loss_poison.append(loss)
                        else:
                            nonmember_loss_poison.append(loss)
                        ##################
                        optimizer.zero_grad()
                        if index == (predictions.size(0)) - 1:
                            loss.backward(retain_graph=False)
                        else:
                            loss.backward(retain_graph=True)
                        # current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])#qu zui hou yi ceng can shu de tidu
                        if self.model_name == "AlexNet":
                            current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])
                        elif self.model_name == "CNNCifar10":
                            current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                        elif self.model_name == "CNNFashion_Mnist":
                            current_grads = model.fc.weight.grad.view([1, 1, 7 * 7 * 32, self.number_of_classes])
                        elif self.model_name == "CNNMnist":
                            current_grads = model.fc2.weight.grad.view([1, 1, 50, self.number_of_classes])
                        elif self.model_name == "LeNet5":
                            current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                        # current_grads = model.lastlayer_name.weight.grad.view([1, 1, 256, self.number_of_classes])
                        elif self.model_name=="CNNModel":
                            current_grads = model.fc2.weight.grad.view([1, 1, 256, self.number_of_classes])
                        else:
                            current_grads = None
                        if gradients.size()[0] == 0:
                            gradients = current_grads
                        else:
                            gradients = torch.cat((gradients, current_grads))
                    # 64, 1, 256, 100 qian 32 ge shi chengyuan zuihouyicengde tidu hou 32shi fei cheng yuan de tidu
                    temp_gradients = torch.from_numpy(gradients.data.cpu().numpy()).to(self.device)
                    loss_labal_poison_list.append(torch.unsqueeze(torch.tensor(loss_labal), dim=1))
                    model_gradients_poison.append(temp_gradients.requires_grad_(True))
                    loss_member_eachbatch_poison.append(member_loss)
                    loss_nomember_eachbatch_poison.append(nonmember_loss)
                    model.to('cpu')



                ####################################################
                ##########ji suan yi ge batch de tidu fan shu
                norm_member_eachbatch = []
                norm_nomember_eachbatch = []

                norm_member_eachbatch_poison = []
                norm_nomember_eachbatch_poison = []

                for model_lastlayer_grads in model_gradients:
                    model_member_grad = model_lastlayer_grads[:segmation]  # qu 0-31

                    model_nonmember_grad = model_lastlayer_grads[segmation:]  # qu 32-63
                    l2_norm_member = torch.norm(model_member_grad, p=2)
                    #for tmp_member in model_member_grad:
                    #    print('l2norm member:',torch.norm(tmp_member, p=2))
                    #for tmp_non_member in model_nonmember_grad:
                    #    print('l2norm non member:',torch.norm(tmp_non_member, p=2))
                    l2_norm_nonmember = torch.norm(model_nonmember_grad, p=2)
                    # fang dao liang ge lie biao
                    # fen bie bao cun mei yi batch de ti du fan shu
                    norm_member_eachbatch.append(l2_norm_member)
                    norm_nomember_eachbatch.append(l2_norm_nonmember)
                l2_norm_member_list.append(norm_member_eachbatch)
                l2_norm_nonmember_list.append(norm_nomember_eachbatch)


                for model_lastlayer_grads in model_gradients_poison:
                    model_member_grad = model_lastlayer_grads[:segmation]  # qu 0-31

                    model_nonmember_grad = model_lastlayer_grads[segmation:]  # qu 32-63
                    l2_norm_member = torch.norm(model_member_grad, p=2)
                    #for tmp_member in model_member_grad:
                    #    print('l2norm member:',torch.norm(tmp_member, p=2))
                    #for tmp_non_member in model_nonmember_grad:
                    #    print('l2norm non member:',torch.norm(tmp_non_member, p=2))
                    l2_norm_nonmember = torch.norm(model_nonmember_grad, p=2)
                    # fang dao liang ge lie biao
                    # fen bie bao cun mei yi batch de ti du fan shu
                    norm_member_eachbatch_poison.append(l2_norm_member)
                    norm_nomember_eachbatch_poison.append(l2_norm_nonmember)
                l2_norm_member_poison_list.append(loss_member_eachbatch_poison)
                l2_norm_nonmember_poison_list.append(norm_nomember_eachbatch_poison)            

                loss_member_eachbatch_avg=[sum(t) / len(t) for t in loss_member_eachbatch]#get each batch avg member loss in each model
                loss_nonmember_eachbatch_avg=[sum(t) / len(t) for t in loss_nomember_eachbatch]#get each batch avg nonmember loss in each model
                loss_member_list.append(loss_member_eachbatch_avg)
                loss_nonmember_list.append(loss_nonmember_eachbatch_avg)


                loss_member_eachbatch_avg_poison=[sum(t) / len(t) for t in loss_member_eachbatch_poison]#get each batch avg member loss in each model
                loss_nonmember_eachbatch_avg_poison=[sum(t) / len(t) for t in loss_nomember_eachbatch_poison]#get each batch avg nonmember loss in each model
                loss_member_poison_list.append(loss_member_eachbatch_avg_poison)
                loss_nonmember_poison_list.append(loss_nonmember_eachbatch_avg_poison)

                diff_loss=[a-b for a,b in zip(loss_labal_ori_list,loss_labal_poison_list)]
                diff_grad=[a-b for a,b in zip(model_gradients,model_gradients_poison)]

                for i in range(len(diff_loss)):
                    diff_loss[i]=diff_loss[i].to(self.device)
                for i in range(len(diff_grad)):
                    diff_grad[i]=diff_grad[i].to(self.device)

                # Get the predictions for the membership classification
                self.attack_model.to(self.device)
                with torch.no_grad():
                    # Get membership predictions
                    membership_predictions = self.attack_model(model_outputs, one_hot_labels, loss_values,
                                                            model_gradients,diff_loss,diff_grad)

                    # Change labels of the data for binary attack classification
                    member_target = torch.Tensor([1 for _ in member_target])
                    non_member_target = torch.Tensor([0 for _ in non_member_target])
                    membership_labels = torch.cat((member_target, non_member_target))
                    membership_labels = torch.unsqueeze(membership_labels, -1).data.float().to(self.device)

                    # Calculate the loss of attack model
                    attack_loss = self.attack_loss_function(membership_predictions, membership_labels)

                    # update class confusion matrices, to see scores per original class
                    combined_labels = zip(labels, membership_labels)
                    for index, (data_class, member_class) in enumerate(combined_labels):
                        class_matrices[data_class.item()].update(membership_predictions[index], member_class)

                    batch_confusion_matrix.update(membership_predictions, membership_labels)
                    accuracy.update(batch_confusion_matrix.get_accuracy(), self.attack_test_batch_size)
                    losses.update(attack_loss.item(), self.attack_test_batch_size)

                self.attack_model.to('cpu')

                temp = batch_confusion_matrix.get_confusion_matrix()
                final_confusion_matrix.update_from_matrix(temp)
                message = f'[ Round: {round_number} ' \
                        f'| Attacker Test ' \
                        f'| Batch Time: {time.time() - batch_start_time:.2f}s ' \
                        f'| Batch Loss: {attack_loss:.5f} ' \
                        f'| Batch Acc: {batch_confusion_matrix.get_accuracy():.2f}% ]' \
                        f'| Class Conf Matrix: TP:{temp[0]}' \
                        f' FP:{temp[1]}' \
                        f' TN:{temp[2]}' \
                        f' FN:{temp[3]} ]'
                logging.info(message)
                if times%self.save_information_count==0:
                    log_attack = logging.getLogger('log_attack')
                    log_attack.info(message)

            # Print the final summary of this test
            temp = final_confusion_matrix.get_confusion_matrix()
            self.best_inference_acc_test=max(self.best_inference_acc_test,final_confusion_matrix.get_accuracy())
            message = f'[ Round: {round_number} Totals ' \
                    f'| Attacker Test ' \
                    f'| Total Time: {time.time() - start_time:.2f}s ' \
                    f'| Loss: {losses.avg:.5f} ' \
                    f'| Acc: {final_confusion_matrix.get_accuracy():.2f}% ' \
                    f'| Best Test Acc: {self.best_inference_acc_test:.2f}% ' \
                    f'| Best Train Acc: {self.best_inference_acc_train:.2f}% ' \
                    f'| Final Conf Matrix: TP:{temp[0]}' \
                    f' FP:{temp[1]}' \
                    f' TN:{temp[2]}' \
                    f' FN:{temp[3]} ]'
            logging.info(message)
            log_attack = logging.getLogger('log_attack')
            log_attack.info(message)

            # Print all the scores for each class.
            message = f'[ Round: {round_number} Class Scores ]\n'
            for data_class in range(self.number_of_classes):
                class_matrix = class_matrices.get(data_class)
                scores = class_matrix.get_confusion_matrix()
                class_message = f'[ Class {data_class}' \
                                f'| Class Acc: {class_matrix.get_accuracy():.2f}% ' \
                                f'| Conf Matrix: TP:{scores[0]}' \
                                f' FP:{scores[1]}' \
                                f' TN:{scores[2]}' \
                                f' FN:{scores[3]} ]'
                if data_class == 99:
                    message += class_message
                else:
                    message += class_message + '\n'
            logging.info(message)
            log_attack = logging.getLogger('log_attack')
            log_attack.info(message)

            return losses.avg, final_confusion_matrix.get_accuracy()


    def test_attack2(self, round_number, target_models):
        """ Ja hier moet dus documentatie """
        # Metrics
        start_time = time.time()
        final_confusion_matrix = ConfusionMatrix()
        batch_confusion_matrix = ConfusionMatrix()
        losses = AverageMeter()
        accuracy = AverageMeter()
        # create dict for keeping track of scores per class.
        class_matrices = dict((data_class, ConfusionMatrix()) for data_class in range(self.number_of_classes))

        self.attack_model.eval()

        member, non_member = self.attack_test_data
        temp_data_loader = DataLoader(member,
                                    batch_size=self.attack_test_batch_size // 2,
                                    shuffle=False,
                                    num_workers=2,
                                    pin_memory=True
                                    )

        temp_data_loader_2 = DataLoader(non_member,
                                        batch_size=self.attack_test_batch_size // 2,
                                        shuffle=False,
                                        num_workers=2,
                                        pin_memory=True
                                        )
        
        member_poison=self.attack_test_member_poison
        non_member_poison=self.attack_test_non_member_poison

        temp_data_loader_poison=DataLoader(member_poison,
                                        batch_size=self.attack_train_batch_size // 2,
                                        shuffle=False,
                                        num_workers=0,
                                        pin_memory=True
        )


        temp_data_loader_2_poison=DataLoader(non_member_poison,
                                        batch_size=self.attack_train_batch_size // 2,
                                        shuffle=False,
                                        num_workers=0,
                                        pin_memory=True
        )

        data_loader = zip(temp_data_loader, temp_data_loader_2,temp_data_loader_poison,temp_data_loader_2_poison)
        #data_loader_poison=zip(temp_data_loader_poison,temp_data_loader_2_poison)
        ####################################################
        # bao cun mei ge batch de ti du fan shu he loss
        l2_norm_member_list = []
        l2_norm_nonmember_list = []
        loss_member_list = []
        loss_nonmember_list = []



        l2_norm_member_poison_list = []
        l2_norm_nonmember_poison_list = []
        loss_member_poison_list = []
        loss_nonmember_poison_list = []


        temp = self.batch_size
        halftbatch = temp // 2  # xu yao geng ju batch zhi tiaozheng!!!
        times = 0
        ####################################################
        for (member_input, member_target), (non_member_input, non_member_target),(member_input_poison, member_target_poison),(non_member_input_poison, non_member_target_poison) in data_loader:
            times+=1
            #
            batch_start_time = time.time()
            batch_confusion_matrix.reset()
            segmation = len(member_input)
            # Pre-define variables to use.
            model_outputs = []
            loss_values = []
            model_gradients = []



            model_outputs_poison = []
            loss_values_poison = []
            model_gradients_poison = []

            #temp_loss_values = []
            # Load data to device
            member_input, member_target = member_input.float().to(self.device) \
                , member_target.long().to(self.device)
            non_member_input, non_member_target = non_member_input.float().to(self.device) \
                , non_member_target.long().to(self.device)
            
            member_input_poison, member_target_poison = member_input_poison.float().to(self.device) \
                , member_target_poison.long().to(self.device)
            non_member_input_poison, non_member_target_poison = non_member_input_poison.float().to(self.device) \
                , non_member_target_poison.long().to(self.device)

            data, labels = torch.cat((member_input, non_member_input)), torch.cat(
                (member_target, non_member_target))

            # Create one-hot encoding of labels, as this has to be done only once.
            one_hot_labels = encode_labels(labels, self.one_hot_encoding, self.device).float()

            data_poison, labels_poison = torch.cat((member_input_poison, non_member_input_poison)), torch.cat(
                (member_target_poison, non_member_target_poison))
            #plt.imshow(data_poison[0].cpu().numpy().transpose(1,2,0))
            #plt.show()
            one_hot_labels_poison = encode_labels(labels_poison, self.one_hot_encoding, self.device).float()
            ####################
            loss_member_eachbatch=[]
            loss_nomember_eachbatch=[]

            loss_member_eachbatch_poison=[]
            loss_nomember_eachbatch_poison=[]
            ####################
            # For each observed model collect input data for the attack model.

            loss_labal_ori_list=[]
            imglist=[]


            for model in target_models:
                tmppp=torch.cat((member_input, non_member_input), dim=0)
                tmppp = Variable(tmppp, requires_grad=True)
                imglist.append(tmppp)
                #
                ####################################################
                # qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                member_loss = []
                nonmember_loss = []
                ####################################################
                model.eval()
                model.to(self.device)
                optimizer = optimizers.__dict__[self.optimizer_name](
                    params=model.parameters(),
                    lr=self.learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay
                )

                # Get predictions
                predictions = model(data)
                temp_pred = torch.from_numpy(predictions.data.cpu().numpy()).to(self.device)
                model_outputs.append(temp_pred.requires_grad_(True))  # ba cheng yuan he fei chen yuan
                # calculate loss values
                # predictions 64*10 one_hot_labels 64*10
                loss_value = torch.sum(predictions * one_hot_labels, dim=1).view([-1, 1])
                ###loss_value是對應標籤的模型輸出值
                temp_loss = torch.from_numpy(loss_value.data.cpu().numpy()).to(self.device)

                loss_values.append(temp_loss.requires_grad_(True))
                loss_labal=[]
                gradients = torch.zeros(0)
                # dui mei yige yangebn qiu loss he tidu
                # qian 32 ge shi cheng yaun hou 32 ge fei cheng yuan
                for index in range(predictions.size(0)):
                    # ji suan yi ge shi ti de loss, qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                    loss = self.loss_function(predictions[index].view([1, -1]), labels[index].view([-1]))
                    loss_labal.append(loss)
                    ################## add loss into list
                    if index < segmation:
                        member_loss.append(loss)
                    else:
                        nonmember_loss.append(loss)
                    ##################
                    optimizer.zero_grad()
                    if index == (predictions.size(0)) - 1:
                        loss.backward(retain_graph=False)
                    else:
                        loss.backward(retain_graph=True)
                    # current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])#qu zui hou yi ceng can shu de tidu
                    if self.model_name == "AlexNet":
                        current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name == "CNNCifar10":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    elif self.model_name == "CNNFashion_Mnist":
                        current_grads = model.fc.weight.grad.view([1, 1, 7 * 7 * 32, self.number_of_classes])
                    elif self.model_name == "CNNMnist":
                        current_grads = model.fc2.weight.grad.view([1, 1, 50, self.number_of_classes])
                    elif self.model_name == "LeNet5":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    # current_grads = model.lastlayer_name.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name=="CNNModel":
                        current_grads = model.fc2.weight.grad.view([1, 1, 256, self.number_of_classes])
                    else:
                        current_grads = None
                    if gradients.size()[0] == 0:
                        gradients = current_grads
                    else:
                        gradients = torch.cat((gradients, current_grads))
                # 64, 1, 256, 100 qian 32 ge shi chengyuan zuihouyicengde tidu hou 32shi fei cheng yuan de tidu
                temp_gradients = torch.from_numpy(gradients.data.cpu().numpy()).to(self.device)
                loss_labal_ori_list.append(torch.unsqueeze(torch.tensor(loss_labal), dim=1))

                model_gradients.append(temp_gradients.requires_grad_(True))
                loss_member_eachbatch.append(member_loss)
                loss_nomember_eachbatch.append(nonmember_loss)
                model.to('cpu')
            



            loss_labal_poison_list=[]
            for model in target_models:
                #
                ####################################################
                # qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                member_loss_poison = []
                nonmember_loss_poison = []
                ####################################################
                model.eval()
                model.to(self.device)
                optimizer = optimizers.__dict__[self.optimizer_name](
                    params=model.parameters(),
                    lr=self.learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay
                )

                # Get predictions
                predictions = model(data_poison)
                _, predicted = torch.max(predictions, 1)

                temp_pred = torch.from_numpy(predictions.data.cpu().numpy()).to(self.device)
                model_outputs_poison.append(temp_pred.requires_grad_(True))  # ba cheng yuan he fei chen yuan
                # calculate loss values
                # predictions 64*10 one_hot_labels 64*10
                loss_value = torch.sum(predictions * one_hot_labels_poison, dim=1).view([-1, 1])

                ###loss_value是對應標籤的模型輸出值
                temp_loss = torch.from_numpy(loss_value.data.cpu().numpy()).to(self.device)

                loss_values_poison.append(temp_loss.requires_grad_(True))

                gradients = torch.zeros(0)
                # dui mei yige yangebn qiu loss he tidu
                # qian 32 ge shi cheng yaun hou 32 ge fei cheng yuan
                loss_labal=[]
                for index in range(predictions.size(0)):
                    # ji suan yi ge shi ti de loss, qian 32 ge cheng yuan hou 32 ge fei cheng yuan
                    loss = self.loss_function(predictions[index].view([1, -1]), labels[index].view([-1]))
                    loss_labal.append(loss)
                    ################## add loss into list
                    if index < segmation:
                        member_loss_poison.append(loss)
                    else:
                        nonmember_loss_poison.append(loss)
                    ##################
                    optimizer.zero_grad()
                    if index == (predictions.size(0)) - 1:
                        loss.backward(retain_graph=False)
                    else:
                        loss.backward(retain_graph=True)
                    # current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])#qu zui hou yi ceng can shu de tidu
                    if self.model_name == "AlexNet":
                        current_grads = model.classifier.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name == "CNNCifar10":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    elif self.model_name == "CNNFashion_Mnist":
                        current_grads = model.fc.weight.grad.view([1, 1, 7 * 7 * 32, self.number_of_classes])
                    elif self.model_name == "CNNMnist":
                        current_grads = model.fc2.weight.grad.view([1, 1, 50, self.number_of_classes])
                    elif self.model_name == "LeNet5":
                        current_grads = model.fc3.weight.grad.view([1, 1, 84, self.number_of_classes])
                    # current_grads = model.lastlayer_name.weight.grad.view([1, 1, 256, self.number_of_classes])
                    elif self.model_name=="CNNModel":
                        current_grads = model.fc2.weight.grad.view([1, 1, 256, self.number_of_classes])
                    else:
                        current_grads = None
                    if gradients.size()[0] == 0:
                        gradients = current_grads
                    else:
                        gradients = torch.cat((gradients, current_grads))
                # 64, 1, 256, 100 qian 32 ge shi chengyuan zuihouyicengde tidu hou 32shi fei cheng yuan de tidu
                temp_gradients = torch.from_numpy(gradients.data.cpu().numpy()).to(self.device)
                loss_labal_poison_list.append(torch.unsqueeze(torch.tensor(loss_labal), dim=1))
                model_gradients_poison.append(temp_gradients.requires_grad_(True))
                loss_member_eachbatch_poison.append(member_loss)
                loss_nomember_eachbatch_poison.append(nonmember_loss)
                model.to('cpu')



            ####################################################
            ##########ji suan yi ge batch de tidu fan shu
            norm_member_eachbatch = []
            norm_nomember_eachbatch = []

            norm_member_eachbatch_poison = []
            norm_nomember_eachbatch_poison = []

            for model_lastlayer_grads in model_gradients:
                model_member_grad = model_lastlayer_grads[:segmation]  # qu 0-31

                model_nonmember_grad = model_lastlayer_grads[segmation:]  # qu 32-63
                l2_norm_member = torch.norm(model_member_grad, p=2)
                #for tmp_member in model_member_grad:
                #    print('l2norm member:',torch.norm(tmp_member, p=2))
                #for tmp_non_member in model_nonmember_grad:
                #    print('l2norm non member:',torch.norm(tmp_non_member, p=2))
                l2_norm_nonmember = torch.norm(model_nonmember_grad, p=2)
                # fang dao liang ge lie biao
                # fen bie bao cun mei yi batch de ti du fan shu
                norm_member_eachbatch.append(l2_norm_member)
                norm_nomember_eachbatch.append(l2_norm_nonmember)
            l2_norm_member_list.append(norm_member_eachbatch)
            l2_norm_nonmember_list.append(norm_nomember_eachbatch)


            for model_lastlayer_grads in model_gradients_poison:
                model_member_grad = model_lastlayer_grads[:segmation]  # qu 0-31

                model_nonmember_grad = model_lastlayer_grads[segmation:]  # qu 32-63
                l2_norm_member = torch.norm(model_member_grad, p=2)
                #for tmp_member in model_member_grad:
                #    print('l2norm member:',torch.norm(tmp_member, p=2))
                #for tmp_non_member in model_nonmember_grad:
                #    print('l2norm non member:',torch.norm(tmp_non_member, p=2))
                l2_norm_nonmember = torch.norm(model_nonmember_grad, p=2)
                # fang dao liang ge lie biao
                # fen bie bao cun mei yi batch de ti du fan shu
                norm_member_eachbatch_poison.append(l2_norm_member)
                norm_nomember_eachbatch_poison.append(l2_norm_nonmember)
            l2_norm_member_poison_list.append(loss_member_eachbatch_poison)
            l2_norm_nonmember_poison_list.append(norm_nomember_eachbatch_poison)            

            loss_member_eachbatch_avg=[sum(t) / len(t) for t in loss_member_eachbatch]#get each batch avg member loss in each model
            loss_nonmember_eachbatch_avg=[sum(t) / len(t) for t in loss_nomember_eachbatch]#get each batch avg nonmember loss in each model
            loss_member_list.append(loss_member_eachbatch_avg)
            loss_nonmember_list.append(loss_nonmember_eachbatch_avg)


            loss_member_eachbatch_avg_poison=[sum(t) / len(t) for t in loss_member_eachbatch_poison]#get each batch avg member loss in each model
            loss_nonmember_eachbatch_avg_poison=[sum(t) / len(t) for t in loss_nomember_eachbatch_poison]#get each batch avg nonmember loss in each model
            loss_member_poison_list.append(loss_member_eachbatch_avg_poison)
            loss_nonmember_poison_list.append(loss_nonmember_eachbatch_avg_poison)

            diff_loss=[a-b for a,b in zip(loss_labal_ori_list,loss_labal_poison_list)]
            diff_grad=[a-b for a,b in zip(model_gradients,model_gradients_poison)]

            for i in range(len(diff_loss)):
                diff_loss[i]=diff_loss[i].to(self.device)
            for i in range(len(diff_grad)):
                diff_grad[i]=diff_grad[i].to(self.device)

            # Get the predictions for the membership classification
            self.attack_model.to(self.device)
            with torch.no_grad():
                # Get membership predictions
                membership_predictions = self.attack_model(model_outputs, one_hot_labels, loss_values,
                                                        model_gradients,diff_loss,diff_grad,imglist)

                # Change labels of the data for binary attack classification
                member_target = torch.Tensor([1 for _ in member_target])
                non_member_target = torch.Tensor([0 for _ in non_member_target])
                membership_labels = torch.cat((member_target, non_member_target))
                membership_labels = torch.unsqueeze(membership_labels, -1).data.float().to(self.device)

                # Calculate the loss of attack model
                attack_loss = self.attack_loss_function(membership_predictions, membership_labels)

                # update class confusion matrices, to see scores per original class
                combined_labels = zip(labels, membership_labels)
                for index, (data_class, member_class) in enumerate(combined_labels):
                    class_matrices[data_class.item()].update(membership_predictions[index], member_class)

                batch_confusion_matrix.update(membership_predictions, membership_labels)
                accuracy.update(batch_confusion_matrix.get_accuracy(), self.attack_test_batch_size)
                losses.update(attack_loss.item(), self.attack_test_batch_size)

            self.attack_model.to('cpu')

            temp = batch_confusion_matrix.get_confusion_matrix()
            final_confusion_matrix.update_from_matrix(temp)
            message = f'[ Round: {round_number} ' \
                    f'| Attacker Test ' \
                    f'| Batch Time: {time.time() - batch_start_time:.2f}s ' \
                    f'| Batch Loss: {attack_loss:.5f} ' \
                    f'| Batch Acc: {batch_confusion_matrix.get_accuracy():.2f}% ]' \
                    f'| Class Conf Matrix: TP:{temp[0]}' \
                    f' FP:{temp[1]}' \
                    f' TN:{temp[2]}' \
                    f' FN:{temp[3]} ]'
            logging.info(message)
            if times%self.save_information_count==0:
                log_attack = logging.getLogger('log_attack')
                log_attack.info(message)

        # Print the final summary of this test
        temp = final_confusion_matrix.get_confusion_matrix()
        self.best_inference_acc_test=max(self.best_inference_acc_test,final_confusion_matrix.get_accuracy())
        message = f'[ Round: {round_number} Totals ' \
                f'| Attacker Test ' \
                f'| Total Time: {time.time() - start_time:.2f}s ' \
                f'| Loss: {losses.avg:.5f} ' \
                f'| Acc: {final_confusion_matrix.get_accuracy():.2f}% ' \
                f'| Best Test Acc: {self.best_inference_acc_test:.2f}% ' \
                f'| Best Train Acc: {self.best_inference_acc_train:.2f}% ' \
                f'| Final Conf Matrix: TP:{temp[0]}' \
                f' FP:{temp[1]}' \
                f' TN:{temp[2]}' \
                f' FN:{temp[3]} ]'
        logging.info(message)
        log_attack = logging.getLogger('log_attack')
        log_attack.info(message)

        # Print all the scores for each class.
        message = f'[ Round: {round_number} Class Scores ]\n'
        for data_class in range(self.number_of_classes):
            class_matrix = class_matrices.get(data_class)
            scores = class_matrix.get_confusion_matrix()
            class_message = f'[ Class {data_class}' \
                            f'| Class Acc: {class_matrix.get_accuracy():.2f}% ' \
                            f'| Conf Matrix: TP:{scores[0]}' \
                            f' FP:{scores[1]}' \
                            f' TN:{scores[2]}' \
                            f' FN:{scores[3]} ]'
            if data_class == 99:
                message += class_message
            else:
                message += class_message + '\n'
        logging.info(message)
        log_attack = logging.getLogger('log_attack')
        log_attack.info(message)

        return losses.avg, final_confusion_matrix.get_accuracy()