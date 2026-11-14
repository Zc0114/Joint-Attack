import copy
import logging
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm
from src.attacker import Attacker
from src.client import Client
from src.models import AlexNet, \
    CNNCifar10, \
    CNNFashion_Mnist, \
    CNNMnist, \
    LeNet5, \
    CNNModel, \
    CNNNet
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



class CentralServer(object):
    """
    Ja hier moet dus documentatie
    """

    def __init__(self, experiment_param, attack_param, data_param, training_param, backdoor_param,imgpath,rootpath):

        """ Ja hier moet dus documentatie """
        self.device = experiment_param['device']
        self.train_model = experiment_param['train_model']
        self.do_local_eval = experiment_param['local_eval']
        self.do_global_eval = experiment_param['global_eval']
        self.save_model = experiment_param['save_model']
        self.model_path = experiment_param['model_path']

        ##############add local_epoch param and other param
        self.local_epoch = training_param['local_epoch']
        self.attack_local_epoch = attack_param['attack_local_epoch']
        self.is_iid = training_param['is_iid']
        self.model_name = experiment_param['model_name']
        self.no_iid_class_num=training_param['no_iid_class_num']
        self.proportion_clients=training_param['proportion_clients']
        self.no_iid_class_num=training_param['no_iid_class_num']
        self.round_list=[]
        #from low_index to high_index and low_index,high_index in do_active_attack
        '''
        low_index=attack_param['active_attack'][0]
        high_index=attack_param['active_attack'][1]
        '''
        self.rootpath=rootpath
        repated=attack_param['active_attack']
        temp_round_num=training_param['training_rounds']
        self.do_active_attack=[]
        for i in range(attack_param['start_index'][0],attack_param['start_index'][1]+1,repated):
            self.do_active_attack.append(i)

        print(self.do_active_attack)
        self.image_size=None
        self.stable_in=attack_param['stable_in']
        self.is_plot=attack_param['is_plot']
        self.chosemaxdiff=attack_param['chosemaxdiff']
        self.other_training=attack_param['other_training']
        self.maxdiff=attack_param['maxdiff']
        ########################################################

        self.attack_param = attack_param
        self.do_passive_attack = attack_param['passive_attack']

        self.save_attack_model = attack_param['save_attack_model']
        self.attack_data_overlap = attack_param['attack_data_target_overlap_with']
        self.observed_target_models = attack_param['observed_target_models']
        self.attack_model_path = attack_param['attack_model_path']
        #self.attack_model_path =experiment_param['model_path']
        self.eval_attack = attack_param['eval_attack']

        self.dataset_path = data_param['data_path']
        self.dataset_name = data_param['dataset_name']
        self.number_of_classes = data_param['number_of_classes']
        self.experiment_param=experiment_param
        self.training_param = training_param

        self.save_img_path=imgpath

        self.number_of_clients = training_param['number_of_clients']
        self.number_of_training_rounds = training_param['training_rounds']
        self.loss_function = get_torch_loss_function(training_param['loss_function'])
        self.adjust_lr_at_ = training_param['adjust_lr_at_']
        self.train_batch_size = training_param['train_batch_size']
        self.test_batch_size = training_param['test_batch_size']
        self.client_data_overlap = training_param['client_data_overlap']
        self.is_dataset_of_fixed_size = training_param['fixed_dataset']
        self.dataset_size = training_param['dataset_size']

        self.results = {"global_loss": [0.0], "global_accuracy": [0.0],"loss_backdoor": [0.0], "accuracy_backdoor": [0.0]}
        self.attack_results = {"loss": [0.0], "accuracy": [0.0]}

        self.test_data = None
        self.global_test_dataloader = None
        self.clients = None
        self.model = None
        self.target_models_for_inference = []

        #################################################################
        self.backdoor_param = backdoor_param
        self.is_sga=backdoor_param['is_sga']
        self.is_train=backdoor_param['is_train']
        self.is_poison = backdoor_param['is_poison']
        self.back_rounds = backdoor_param['back_rounds']
        self.back_prop = backdoor_param['back_prop']
        self.test_back_prop = backdoor_param['test_back_prop']
        self.noise_percentage = backdoor_param['noise_percentage']
        self.probability_threshold = backdoor_param['probability_threshold']
        self.max_iteration = backdoor_param['max_iteration']
        self.target = backdoor_param['target']
        self.patch_lr = backdoor_param['patch_lr']
        self.patch_type = backdoor_param['patch_type']
        self.log_dir = backdoor_param['log_dir']

        self.adv_patch_cpu=None
        self.test_data_backdoor=None
        self.is_gradient_ascent=attack_param['is_gradient_ascent']
        self.poison_dataset=attack_param['poison_dataset']
        self.normal_training=attack_param['normal_training']
        self.backdoor_way=attack_param['backdoor_way']
        #################################################################

        ##################
        # add model name info
        ##################
        message = f"[ Main Settings: \n" \
                  f"| Device: {self.device} \n" \
                  f"| Train Model: {self.train_model} \n" \
                  f"| global_eval: {self.do_global_eval} \n" \
                  f"| save_model: {self.save_model} \n" \
                  f"| clients: {self.number_of_clients} \n" \
                  f"| classes: {self.number_of_classes} \n" \
                  f"| local learning rate: {training_param['learning_rate']} \n" \
                  f"| local optimizer: {training_param['optimizer']} \n" \
                  f"| learning_rate_schedule: {self.adjust_lr_at_} \n" \
                  f"| train_batch_size: {self.train_batch_size} \n" \
                  f"| test_batch_siez: {self.test_batch_size} \n" \
                  f"| rounds: {self.number_of_training_rounds} \n" \
                  f"| overlap: {self.client_data_overlap} \n" \
                  f"| fixed_dataset: {self.is_dataset_of_fixed_size} \n" \
                  f"| fixed_dataset_size: {self.dataset_size} \n" \
                  f"| passive_attack: {self.do_passive_attack} \n" \
                  f"| active_attack: {self.do_active_attack} \n" \
                  f"| eval_attack: {self.eval_attack} \n" \
                  f"| save_attack_model: {self.save_attack_model} \n" \
                  f"| observed_target_models: {self.observed_target_models} \n" \
                  f"| model_path: {self.model_path} \n" \
                  f"| model_name: {self.model_name} \n" \
                  f"| Attack_model_path: {self.attack_model_path} ]\n"
        logging.info(msg=message)
        log_global = logging.getLogger('log_global')
        log_global.info(message)

    def start_up(self):
        """ Ja hier moet dus documentatie """
        if self.model_name == "AlexNet":
            self.model = AlexNet(self.number_of_classes)
        elif self.model_name == "CNNCifar10":
            self.model = CNNCifar10(self.number_of_classes)
        elif self.model_name == "CNNFashion_Mnist":
            self.model = CNNFashion_Mnist(self.number_of_classes)
        elif self.model_name == "CNNMnist":
            self.model = CNNMnist(self.number_of_classes)
        elif self.model_name == "LeNet5":
            self.model = LeNet5(self.number_of_classes)
        elif self.model_name=="CNNModel":
            self.model=CNNModel()
        elif self.model_name=='CNNNet':
            self.model = CNNNet(self.number_of_classes)

        ###################################################################################
        # Initialize the patch
        data_set_name = self.dataset_name
        if data_set_name == 'tiny-imagenet-100':
            self.image_size = (3, 32, 32)
        elif data_set_name == 'cifar-100-python':
            self.image_size = (3, 32, 32)
        elif data_set_name == "cifar10":
            self.image_size = (3, 32, 32)
        elif data_set_name == "mnist":
            self.image_size = (1, 28, 28)
        elif data_set_name == "fashion_mnist":
            self.image_size = (1, 28, 28)
        self.adv_patch_cpu = patch_initialization(self.patch_type, image_size=self.image_size,
                                                  noise_percentage=self.noise_percentage)
        ###################################################################################

        self.clients = self.generate_clients(self.experiment_param,self.training_param, self.attack_param, self.backdoor_param)
        self.load_datasets(self.dataset_name,
                           self.dataset_path,
                           self.number_of_clients,
                           self.is_iid
                           )



        self.share_model_with_clients()
        '''
        if self.do_passive_attack > 0:
            for index in self.observed_target_models:
                filename = f'{self.model_name}_epoch_{index}_main_clients_{self.number_of_clients}_batch_{self.train_batch_size}'
                self.target_models_for_inference.append(
                    load_target_model(self.model_path, filename, self.model_name, self.number_of_classes))
        '''
        message = 'Completed main server startup'
        logging.debug(message)

        log_local = logging.getLogger('log_local')
        log_local.info(message)

        log_global = logging.getLogger('log_global')
        log_global.info(message)

        log_attack = logging.getLogger('log_attack')
        log_attack.info(message)

    def generate_clients(self,experiment_param, training_param, attack_param,backdoor_param):
        """ Ja hier moet dus documentatie """
        attacker_is_generated = False
        clients = []
        for client_id, dataset in enumerate(range(self.number_of_clients)):
            if (self.do_passive_attack > 0 or self.do_active_attack != []) and not attacker_is_generated:
                clients.append(Attacker(client_id + 1,
                                        experiment_param,
                                        training_param,
                                        attack_param,
                                        backdoor_param,
                                        device=self.device,
                                        target_train_model=copy.deepcopy(self.model),
                                        number_of_observed_models=len(self.observed_target_models),
                                        model_name=self.model_name,
                                        number_of_classes=self.number_of_classes,
                                        data_name=self.dataset_name,
                                        adv_patch_cpu=self.adv_patch_cpu
                                        # number_of_observed_models=1
                                        )
                               )
                attacker_is_generated = True
            else:
                clients.append(Client(client_id + 1,
                                      training_param,
                                      device=self.device,
                                      model=copy.deepcopy(self.model)))

        message = f"Created {len(clients)} clients!"
        logging.debug(message)
        log_local = logging.getLogger('log_local')
        log_local.info(message)

        log_global = logging.getLogger('log_global')
        log_global.info(message)
        log_attack = logging.getLogger('log_attack')
        log_attack.info(message)
        return clients

    def load_datasets(self, data_name, data_path, number_of_clients,is_iid):
        """ Ja hier moet dus documentatie """
        # load correct dataset from torchvision
        train_data, test_data = load_dataset(data_path, data_name)
        self.global_test_dataloader = DataLoader(test_data,
                                                 batch_size=self.test_batch_size,
                                                 shuffle=False,
                                                 pin_memory=True)

        #############################################
        self.test_data_backdoor = test_data
        #############################################

        ##################################sort dataset by label
        train_data_sort_indices = sorted(range(len(train_data)), key=lambda i: train_data[i][1])  # 按标签排序的索引
        train_data_sort = Subset(train_data, train_data_sort_indices)
        ##################################

        # gebruik list om aantal per class te weten
        if self.is_dataset_of_fixed_size == 1:
            data_size = self.dataset_size
        else:
            data_size = len(train_data) // self.number_of_clients
        # print("data_size",data_size)  # 12000

        client_choice_lists = []
        if is_iid == 1:
            # repated selection
            if self.client_data_overlap == 1:
                train_set = range(len(train_data))
                for client in range(number_of_clients):
                    choices = np.random.choice(train_set, size=data_size, replace=True)
                    client_choice_lists.append(list(choices))
                    client_subset = torch.utils.data.Subset(train_data, choices)
                    self.clients[client].load_data(client_subset)
            else:
                client_indices = []
                for _ in range(self.number_of_clients):
                    client_indices.append([])

                for data_class in range(self.number_of_classes):
                    # Set the boundaries for picking samples from the right classes
                    # low high need change by dataset
                    # eachnumber_of_classes
                    ########################
                    eachnumber_of_classes = len(train_data) // self.number_of_classes #每个类的数量
                    # print("eachnumber_of_classes",eachnumber_of_classes) #6000
                    ########################
                    low, high = data_class * eachnumber_of_classes, (data_class + 1) * eachnumber_of_classes
                    random_subset = list(range(low, high))
                    random.shuffle(random_subset)
                    length = data_size // self.number_of_classes  # each class num
                    for _ in range(length):
                        for client_list in client_indices:
                            client_list.append(random_subset.pop())

                # from each class select to client
                for client in range(number_of_clients):
                    choices = client_indices[client]
                    client_choice_lists.append(list(choices))
                    client_subset = torch.utils.data.Subset(train_data_sort, choices)
                    self.clients[client].load_data(client_subset)
        else:
            '''
            if data_name == 'cifar-100-python':
                dict_users=cifar_100_noniid(train_data,self.number_of_clients)
                data_size=len(dict_users[0])
                for client in range(number_of_clients):
                    choices=dict_users[client]
                    client_choice_lists.append(list(choices))
                    client_subset = torch.utils.data.Subset(train_data, choices)
                    self.clients[client].load_data(client_subset)
            elif data_name=='cifar10':
                dict_users = cifar_10_noniid(train_data, self.number_of_clients)
                data_size = len(dict_users[0])
                for client in range(number_of_clients):
                    choices = dict_users[client]
                    client_choice_lists.append(list(choices))
                    client_subset = torch.utils.data.Subset(train_data, choices)
                    self.clients[client].load_data(client_subset)
            elif data_name=='mnist':
                dict_users = mnist_noniid(train_data, self.number_of_clients)
                data_size = len(dict_users[0])
                for client in range(number_of_clients):
                    choices = dict_users[client]
                    client_choice_lists.append(list(choices))
                    client_subset = torch.utils.data.Subset(train_data, choices)
                    self.clients[client].load_data(client_subset)
            elif data_name=='fashion_mnist':
                dict_users = fashion_mnist_noniid(train_data, self.number_of_clients)
                data_size = len(dict_users[0])
                for client in range(number_of_clients):
                    choices = dict_users[client]
                    client_choice_lists.append(list(choices))
                    client_subset = torch.utils.data.Subset(train_data, choices)
                    self.clients[client].load_data(client_subset)
            '''
            ######################################################## no iid data
            dict_users = get_noniid_datadict(data_name,train_data, self.number_of_clients, self.number_of_classes,self.client_data_overlap,self.no_iid_class_num)
            ########################################################
            data_size = len(dict_users[0])
            # print("data_size",data_size) # 12000

            for client in range(number_of_clients):
                choices = dict_users[client]
                client_choice_lists.append(list(choices))
                client_subset = torch.utils.data.Subset(train_data, choices)
                self.clients[client].load_data(client_subset)

        # print(np.shape(client_choice_lists)) #(10, 6000)

        if self.do_passive_attack > 0 or self.do_active_attack != []:
            # If the condition below is met, sample member data from all clients
            if isinstance(self.attack_data_overlap, str) and self.attack_data_overlap == 'all':
                # Send data files to attacker for inference
                    # self.clients[0].load_attack_data(train_data, test_data)
                attack_data_indices = []
                attack_data_indices_others=[]
                for index in range(number_of_clients):
                    if index!=0:
                        attack_data_indices_others+=client_choice_lists[index]
                    attack_data_indices += client_choice_lists[index]
                if self.do_passive_attack > 0:
                    self.clients[0].load_attack_data(train_data, test_data, attack_data_indices, data_size,self.rootpath,0)
                else:
                    self.clients[0].load_attack_data(train_data, test_data, attack_data_indices, data_size,self.rootpath,0)

            else:
                # Only provide member samples for 1 or more clients.
                attack_data_indices = []
                for index in range(self.attack_data_overlap):
                    attack_data_indices += client_choice_lists[index]
                self.clients[0].load_attack_data(train_data, test_data, attack_data_indices, data_size,self.rootpath)

        message = f"Dataset loaded and divided into {data_size} samples per client"
        ######################################################## save infomation
        if self.client_data_overlap == 1:
            message += ' with overlap '
        else:
            message += ' with no overlap '
        logging.info(message)
        log_local = logging.getLogger('log_local')

        log_local.info(message)

        log_global = logging.getLogger('log_global')
        log_global.info(message)
        message = 'Distributed data among clients'
        logging.debug(message)
        log_local = logging.getLogger('log_local')

        log_local.info(message)

        log_global = logging.getLogger('log_global')
        log_global.info(message)
        ########################################################
    def share_model_with_clients(self):
        """ Ja hier moet dus documentatie """
        for client in self.clients:
            client.model = copy.deepcopy(self.model)
        message = 'Shared model with clients...'
        logging.debug(message)
        log_local = logging.getLogger('log_local')

        log_local.info(message)
        log_global = logging.getLogger('log_global')
        log_global.info(message)

        log_attack = logging.getLogger('log_attack')
        log_attack.info(message)
    def aggregate_model(self):
        """ Ja hier moet dus documentatie """
        # print("Aggregating models....")
        averaged_dict = self.model.state_dict()

        for i in self.round_list[-1]:
            self.clients[i].model.to(self.device)
        self.model.to(self.device)
        #self.round_list.append(idxs_users)
        #tmp_clients=[self.clients[i] for i in self.round_list[-1]]
        for layer in averaged_dict.keys():
            averaged_dict[layer] = torch.stack(
                [self.clients[i].model.state_dict()[layer].float() for i in self.round_list[-1]], 0).mean(0)
        self.model.load_state_dict(averaged_dict)
        message = 'Aggregated model'
        logging.debug(message)
        log_local = logging.getLogger('log_local')

        log_local.info(message)

        log_global = logging.getLogger('log_global')
        log_global.info(message)

        log_attack = logging.getLogger('log_attack')
        log_attack.info(message)

    def do_training(self, round_number):
        """ Ja hier moet dus documentatie """
        # Do client training
        message='client traing'

        log_global = logging.getLogger('log_global')
        log_global.info(message)
        log_attack = logging.getLogger('log_attack')
        log_attack.info(message)
        ##################################################################select clients
        if self.stable_in==1:
            m = max(int(self.proportion_clients * self.number_of_clients)-1, 1)
            idxs_users = np.random.choice(range(1,self.number_of_clients), m, replace=False)
            idxs_users = np.append(idxs_users, 0)
            self.round_list.append(idxs_users)
        else:
            m = max(int(self.proportion_clients * self.number_of_clients), 1)
            idxs_users = np.random.choice(range(0, self.number_of_clients), m, replace=False)
            self.round_list.append(idxs_users)

        tmp_clients = [self.clients[i] for i in idxs_users]
        ##################################################################

        ##################################################################
        for client in tmp_clients:
            if client.client_id==1:
                if self.normal_training==1:
                    client.train(round_number, self.local_epoch)
                    if self.do_local_eval > 0 and round_number % self.do_local_eval == 0:
                        client.test(round_number)
                if self.other_training > 0 and round_number not in self.do_active_attack:
                    self.clients[0].other_training(round_number, self.local_epoch)

                #else:

            else:
                # print("client", client)
                client.train(round_number,self.local_epoch)
                if self.do_local_eval > 0 and round_number % self.do_local_eval == 0:
                    client.test(round_number)
        ##################################################################

    def test_global_model(self, round_number):
        """ Ja hier moet dus documentatie """
        self.model.eval()
        self.model.to(self.device)

        batch_time = AverageMeter()
        losses = AverageMeter()

        data_size = len(self.global_test_dataloader.dataset)
        correct = 0

        message = f'[ Round: {round_number} | Global Model Eval started ]'
        logging.info(message)
        log_global = logging.getLogger('log_global')
        log_global.info(message)
        with torch.no_grad():
            start_time = time.time()
            for data, labels in self.global_test_dataloader:
                # Transfer data to device
                data, labels = data.float().to(self.device), labels.long().to(self.device)

                # Do a forward pass through the network to get prediction values and update loss metric.
                outputs = self.model(data)
                losses.update(self.loss_function(outputs, labels))

                # Compare predictions to labels and get accuracy score.
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                # Update loss, accuracy and run_time metrics
                batch_time.update(time.time() - start_time)

            # Create and log message about test
            accuracy = (correct / data_size) * 100
            message = f'[ Round: {round_number} ' \
                      f'| Loss: {losses.avg:.5f}' \
                      f'| Accuracy: ({correct}/{data_size})={accuracy:.2f}% ]'

            logging.info(message)
            log_global = logging.getLogger('log_global')
            log_global.info(message)
        self.model.to("cpu")

        return losses.avg, accuracy


    def test_global_model_backdoor(self, round_number):
        """ Ja hier moet dus documentatie """
        self.model.eval()
        self.model.to(self.device)

        batch_time = AverageMeter()
        losses_backdoor = AverageMeter()

        ###################################################################################
        print("test poison poison")
        # print("self.adv_patch_cpu",self.adv_patch_cpu)

        if self.dataset_name == 'mnist' or self.dataset_name == 'fashion_mnist':
            dataset_poison = MyDataset_mnist(self.adv_patch_cpu, self.test_data_backdoor, self.target, self.test_back_prop, mode="test")
        else:
            dataset_poison = MyDataset_cifar_new(self.adv_patch_cpu, self.test_data_backdoor,
                                                 self.target, self.back_prop, mode="train",
                                                 number_of_classes=self.number_of_classes)

        global_test_dataloade_backdoor = DataLoader(dataset_poison,
                                        batch_size=self.test_batch_size,
                                        shuffle=False,
                                        pin_memory=True)
        ###################################################################################
        data_size = len(global_test_dataloade_backdoor.dataset)
        correct = 0

        message = f'[ Round: {round_number} | Global Backdoor Model Eval started ]'
        logging.info(message)

        log_global = logging.getLogger('log_global')
        log_global.info(message)

        log_attack = logging.getLogger('log_attack')
        log_attack.info(message)
        with torch.no_grad():
            start_time = time.time()
            for data, labels in global_test_dataloade_backdoor:
                # Transfer data to device
                data, labels = data.float().to(self.device), labels.long().to(self.device)

                # Do a forward pass through the network to get prediction values and update loss metric.
                outputs = self.model(data)
                losses_backdoor.update(self.loss_function(outputs, labels))

                # Compare predictions to labels and get accuracy score.
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                # Update loss, accuracy and run_time metrics
                batch_time.update(time.time() - start_time)

            # Create and log message about test
            accuracy_backdoor = (correct / data_size) * 100
            message = f'[ Round: {round_number} ' \
                      f'| Loss: {losses_backdoor.avg:.5f}' \
                      f'| Backdoot Test Accuracy: ({correct}/{data_size})={accuracy_backdoor:.2f}% ]'
            logging.info(message)
            log_global = logging.getLogger('log_global')
            log_global.info(message)

            log_attack = logging.getLogger('log_attack')
            log_attack.info(message)

        self.model.to("cpu")

        return losses_backdoor.avg, accuracy_backdoor


    def plot_global(self):
        print('ploting global')

        
        global_loss = self.results['global_loss']
        global_accuracy = self.results['global_accuracy']
        x = [i for i in range(len(global_loss))]
        #print(global_loss)
        #print(global_accuracy)
        plt.figure()
        plt.plot(x, global_loss, label='global_loss-round')
        plt.xlabel('round')
        plt.ylabel('global_loss')
        with open('./log/persistent.txt', 'r', encoding="utf-8") as f:
            read_data = int(f.read())
            file_number = str(read_data + 1)

        model_path_split = self.rootpath.split('/')
        #print(model_path_split)
        name = 'globalloss-' + model_path_split[-2]
        plt.title(name)
        name = self.save_img_path + name + file_number + '.png'
        #print(name)
        plt.legend()  # 添加图例
        plt.savefig(name)

        plt.figure()
        plt.plot(x, global_accuracy, label='global_accuracy-round')
        plt.xlabel('round')
        plt.ylabel('global_accuracy')
        with open('./log/persistent.txt', 'r', encoding="utf-8") as f:
            read_data = int(f.read())
            file_number = str(read_data + 1)

        name = 'globalaccuracy-' + model_path_split[-2]
        plt.title(name)
        name = self.save_img_path + name + file_number + '.png'
        plt.legend()  # 添加图例
        plt.savefig(name)

    def plot_backdoor(self):
        print('ploting backdoor')
        
        loss_backdoor = self.results['loss_backdoor']
        accuracy_backdoor = self.results['accuracy_backdoor']
        x = [i for i in range(len(loss_backdoor))]
        #print(loss_backdoor)
        #print(accuracy_backdoor)
        plt.figure()
        plt.plot(x, loss_backdoor, label='loss_backdoor-round')
        plt.xlabel('round')
        plt.ylabel('loss_backdoor')

        with open('./log/persistent.txt', 'r', encoding="utf-8") as f:
            read_data = int(f.read())
            file_number = str(read_data + 1)
        model_path_split = self.rootpath.split('/')
        name = 'backdoorloss-' + model_path_split[-2]
        plt.title(name)
        name = self.save_img_path + name + file_number + '.png'

        plt.legend()  # 添加图例
        plt.savefig(name)

        plt.figure()
        plt.plot(x, accuracy_backdoor, label='accuracy_backdoor-round')
        plt.xlabel('round')
        plt.ylabel('accuracy_backdoor')
        with open('./log/persistent.txt', 'r', encoding="utf-8") as f:
            read_data = int(f.read())
            file_number = str(read_data + 1)


        name='backdooraccuracy-'+model_path_split[-2]
        plt.title(name)
        name = self.save_img_path +name +file_number + '.png'

        plt.legend()  # 添加图例
        plt.savefig(name)

    def plot_membership(self):
        print('ploting membership')
        membership_loss = self.attack_results['loss']
        membership_accurary = self.attack_results['accuracy']
        x = [i for i in range(len(membership_accurary))]

        plt.figure()
        plt.plot(x, membership_loss, label='membership_loss-round')
        plt.xlabel('round')
        plt.ylabel('membership_loss')


        with open('./log/persistent.txt', 'r', encoding="utf-8") as f:
            read_data = int(f.read())
            file_number = str(read_data + 1)
        model_path_split=self.rootpath.split('/')
        name='membershiploss-'+model_path_split[-2]
        plt.title(name)
        name =self.save_img_path+name +file_number + '.png'
        plt.legend()  # 添加图例
        plt.savefig(name)

        plt.figure()
        plt.plot(x, membership_accurary, label='membership_accurary-round')
        plt.xlabel('round')
        plt.ylabel('membership_accurary')
        with open('./log/persistent.txt', 'r', encoding="utf-8") as f:
            read_data = int(f.read())
            file_number = str(read_data + 1)

        name = 'membershipaccurary-' + model_path_split[-2]
        plt.title(name)
        name = self.save_img_path + name + file_number + '.png'

        plt.legend()  # 添加图例
        plt.savefig(name)
        self.clients[0]







    def perform_experiment(self):
        """ Ja hier moet dus documentatie """
        round_loss, round_accuracy = 0, 0
        attacker = self.clients[0]
        target_models_for_inference=[]

        if self.is_plot == 1:
            for index in self.observed_target_models:
                filename = f'{self.model_name}_epoch_{index}_main_clients_{self.number_of_clients}_batch_{self.train_batch_size}'
                target_models_for_inference.append(
                    load_target_model(self.model_path, filename, self.model_name, self.number_of_classes))
            attacker.plot_img(target_models_for_inference)


        if self.chosemaxdiff==1:



            start,end,jiange=self.maxdiff
            observed_target_models=[i for i in range(start,end+1,jiange)]

            message = f'[ Chose max started!  ' \
                      f'| Observed target models: {observed_target_models} ' \



            log_attack = logging.getLogger('log_attack')
            log_attack.info(message)


            print(observed_target_models)
            target_models_for_inference=[]
            for index in observed_target_models:
                filename = f'{self.model_name}_epoch_{index}_main_clients_{self.number_of_clients}_batch_{self.train_batch_size}'
                target_models_for_inference.append(
                    load_target_model(self.model_path, filename, self.model_name, self.number_of_classes))

            target_models_for_ori = []
            model_path='./cifar10/newchangedatadis/cifar10_10_1_gradup_-5-5-5-a/global_model/'
            for index in observed_target_models:
                filename = f'{self.model_name}_epoch_{index}_main_clients_{self.number_of_clients}_batch_{self.train_batch_size}'
                target_models_for_ori.append(
                    load_target_model(model_path, filename, self.model_name, self.number_of_classes))

            attacker.chose_max_epoch(target_models_for_inference,target_models_for_ori,self.save_img_path)


        attacker.genactivatedataloader(self.clients[0].training_data)
        for index in tqdm(range(self.number_of_training_rounds)):
            print("index",index+1)

            index += 1


            #if index == self.number_of_training_rounds-1:
            #    attacker.plot_backdoor_img()
            ########################################################save infomation
            message = f'[ Round: {index} | Started! ]'
            logging.info(message)
            log_local = logging.getLogger('log_local')
            log_local.info(message)

            log_global = logging.getLogger('log_global')
            log_global.info(message)
            log_attack = logging.getLogger('log_attack')
            log_attack.info(message)
            ########################################################
            # If checked, do training cycle
            if self.train_model > 0 and index % self.train_model == 0:
                # Adjust the learning rates at given epochs
                if index in self.adjust_lr_at_:
                    for client in self.clients:
                        client.learning_rate *= 0.1

                self.do_training(index)

                # If checked, perform gradient ascent learning on the attacker dataset each round.
                if index in self.do_active_attack and self.do_active_attack != []:
                    if 0 in self.round_list[-1]:
                        ##########################if jug
                        if self.is_gradient_ascent>0:
                            attacker.gradient_ascent_attack(index,self.attack_local_epoch)
                        ###################################################
                        if self.is_poison > 0:
                            #sadadsasdas=1
                            #attacker.backdoor_attack(index,self.attack_local_epoch)
                            if self.is_sga==0:
                                is_sga=0
                            else:
                                pos=self.do_active_attack.index(index)+1
                                if pos %self.is_sga==0:
                                    is_sga=1
                                else:
                                    is_sga=0

                            if self.backdoor_way==0:
                                self.adv_patch_cpu=attacker.backdoor_attack_0(index,is_sga,self.is_train,self.image_size)
                            else:
                                self.adv_patch_cpu=attacker.backdoor_attack_1(index,is_sga,self.is_train,self.image_size)
                        ###################################################

                        ###################################################


                self.aggregate_model()
                self.share_model_with_clients()
                attacker.get_member_nomember_infomation()
                # If checked, perform global model evaluation every round.
                if self.do_global_eval > 0 and index % self.do_global_eval == 0:
                    round_loss, round_accuracy = self.test_global_model(index)
                    self.results['global_loss'].append(round_loss.cpu().item())
                    self.results['global_accuracy'].append(round_accuracy)
                    #####################################################################################
                    #each round run test backdoor
                    if self.is_poison>0:
                        round_loss_backdoor, round_accuracy_backdoor = self.test_global_model_backdoor(index)
                        self.results['loss_backdoor'].append(round_loss_backdoor.cpu().item())
                        self.results['accuracy_backdoor'].append(round_accuracy_backdoor)
                    #####################################################################################
                    #####################################################################################
                    #may need this part in normal
                    #if 0 in self.round_list[-1]:
                    #    if self.is_poison>0:
                    #        round_loss_backdoor, round_accuracy_backdoor = self.test_global_model_backdoor(index)
                    #        self.results['loss_backdoor'].append(round_loss_backdoor)
                    #        self.results['accuracy_backdoor'].append(round_accuracy_backdoor)
                    #####################################################################################

                # If checked, save the current model and optimizer state
                saveflag=True
                if (self.is_poison>0 or self.is_gradient_ascent>0):
                    if index<self.do_active_attack[0]:
                        saveflag=False

                if self.save_model > 0 and index % self.save_model == 0 and saveflag==True:
                    is_best = (round_accuracy >= max(self.results['global_accuracy']))
                    save_checkpoint({
                        'epoch': index,
                        'state_dict': self.model.state_dict(),
                        'acc': round_accuracy,
                        'best_acc': is_best
                    }, is_best=is_best,
                        filename=f'{self.model_name}'
                                 f'_epoch_{index}'
                                 f'_main_clients_{self.number_of_clients}'
                                 f'_batch_{self.train_batch_size}',
                        checkpoint=self.model_path
                    )


            if self.train_model > 0 and index==self.number_of_training_rounds and self.is_poison==0:
                round_loss_backdoor, round_accuracy_backdoor = self.test_global_model_backdoor(index)
                message = f'[ Round: {index} | Finished! ' \
                          f'| Backdoor Loss:{round_loss_backdoor:.5f}' \
                          f'| Backdoor Accuracy: {round_accuracy_backdoor:.5f}]'
                logging.info(message)
                log_global = logging.getLogger('log_global')
                log_global.info(message)
            # If checked, perform passive MIA during each round.
            if self.do_passive_attack > 0 and index % self.do_passive_attack == 0 and index>self.do_active_attack[0]:
                temp_target_models_for_inference=[]
                temp_target_models_for_inference.append(self.model)
                attacker.train_attack(index, temp_target_models_for_inference) 
                #attacker.test_attack(index, temp_target_models_for_inference)
                # if self.is_train==1:
                #     attacker.train_trigger(index, temp_target_models_for_inference,lr=1,way=0)
                #     attacker.regenpoisondata()
                #attacker.train_trigger(index, temp_target_models_for_inference,lr=0.1,way=1)
                #attacker.regenpoisondata()
                #attacker.train_trigger(index, temp_target_models_for_inference,lr=1,way=1)
                #attacker.regenpoisondata()
                #attacker.train_trigger(index, temp_target_models_for_inference,lr=1,way=1)
                #attacker.regenpoisondata()

                if self.eval_attack > 0 and index % self.eval_attack == 0:
                    round_loss, round_accuracy = attacker.test_attack(index, temp_target_models_for_inference)
                    self.attack_results['loss'].append(round_loss)
                    self.attack_results['accuracy'].append(round_accuracy)
                ########################################################save model
                if self.save_attack_model > 0 and index % self.save_attack_model == 0:
                    is_best = (round_accuracy >= max(self.attack_results['accuracy']))
                    if is_best:
                        save_checkpoint_adversary({
                            'model_name': self.model_name,
                            'epoch': index,
                            'state_dict': self.clients[0].model.state_dict(),
                            'acc': round_accuracy,
                            'best_acc': is_best,
                            'optimizer': attacker.attack_optimizer.state_dict()
                        }, is_best=is_best,
                            filename=f'{self.model_name}'
                                     f'_attack_clients_{self.number_of_clients}'
                                     f'_epoch_{index}'
                                     f'_batch_{attacker.attack_test_batch_size}',
                            checkpoint=self.attack_model_path
                        )
                    ########################################################save model


            message = f'[ Round: {index} | Finished! ]'
            logging.info(message)
            log_local = logging.getLogger('log_local')
            log_local.info(message)

            log_global = logging.getLogger('log_global')
            log_global.info(message)


        #attacker.plot_backdoor_img()
        if self.do_passive_attack == 1:
            self.plot_membership()
            #attacker.plot_tsne(self.target_models_for_inference,self.model_path)

        if self.is_poison == 1:
            self.plot_backdoor()
            trigger_path=self.rootpath+'/trigger/'
            if not os.path.exists(trigger_path):
                os.makedirs(trigger_path)
            attacker.save_patch(trigger_path,self.do_active_attack)


        if self.train_model == 1:
            self.plot_global()