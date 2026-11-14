import time
import logging
import torch.utils
import torch.optim as optimizers
from typing import Iterable, Optional
import copy
from torch.nn.utils.convert_parameters import _check_param_device
from torch.utils.data import DataLoader
from src.utils import AverageMeter, get_torch_loss_function
import numpy as np

class Client(object):
    """
     Client Class.

     Used to simulate a client in the Federated Learning network.
     Methods are for loading data, getting and setting the model and
     training and testing on local data with a local model.

     """

    def __init__(self, client_id, training_param, device, model):
        """
        Client constructor method.

        Parameters:
            client_id: Used to identify the client.
            training_param: Training parameters containing setting from the config.yaml file.
            device: Device to be used when doing training and testing (CPU or Cuda GPU).
            model: Copy of the global model used to train on the local data.
        """
        self.client_id = client_id
        self.device = device
        self.__model = model
        # training parameters.
        self.loss_function = get_torch_loss_function(training_param['loss_function'])
        self.batch_size = training_param['train_batch_size']
        # Optimizer parameters.
        self.optimizer_name = training_param['optimizer']
        self.learning_rate = training_param['learning_rate']
        self.momentum = training_param['momentum']
        self.weight_decay = training_param['weight_decay']
        # class shared variables used by methods are initialized here.
        self.training_data = None
        self.test_data = None
        self.training_dataloader = None
        self.testing_dataloader = None
        # scores dictionary.
        self.local_results = {"loss": [], "accuracy": []}

    @property
    def model(self):
        """
         Getter method for client model.
         """
        return self.__model

    @model.setter
    def model(self, model):
        """
         Setter method for client model, also initializes,
         the optimizer with the new model parameters.
         """
        if self.optimizer_name=='SGD':
            self.optimizer = optimizers.__dict__[self.optimizer_name](
                params=model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        else:
            self.optimizer = optimizers.__dict__[self.optimizer_name](
                params=model.parameters(),
                lr=self.learning_rate,
                #momentum=self.momentum,
                #weight_decay=self.weight_decay
            )
        self.__model = model

    def load_data(self, training_data):
        """
         Method for loading the local data set from the main server.
         Creates a DataLoader object for training and testing.
         """
        self.training_data = training_data
        self.training_dataloader = DataLoader(self.training_data,
                                              batch_size=self.batch_size,
                                              shuffle=True,
                                              pin_memory=True)


        message = f'Client {self.client_id} loaded datasets successfully'
        logging.debug(message)

        log_local = logging.getLogger('log_local')

        log_local.info(message)



    def train(self, round_number, local_epoch):
        """
         Training method for the client class.

         Parameters:
             round_number: number of the current epoch to be used for logging.
         """
        # Set model to training mode and transfer to device.
        self.model.train()
        self.model.to(self.device)

        # load once to save time. To be used for calculating the accuracy.
        data_size = len(self.training_dataloader.dataset)
        # print("data_size",data_size) # 6000

        # Define attributes to keep track of scores during training
        batch_time = AverageMeter()
        losses = AverageMeter()
        start_time = time.time()
        correct = 0

        ##################more epoch
        for epoch in range(local_epoch):
        #################################
            for data, labels in self.training_dataloader:
                # Transfer data to CPU or GPU and set gradients to zero for performance.
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                self.optimizer.zero_grad()

                # Do a forward pass through the network to get prediction values and update loss metric.
                outputs = self.model(data)
                loss = self.loss_function(outputs, labels)

                # Do a backward pass through the network to get the gradients
                # and then use the optimizer to update the weights.
                loss.backward()
                self.optimizer.step()

                # Compare predictions to labels and get accuracy score.
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                # Update loss, accuracy and run_time metrics
                losses.update(loss.item())
                batch_time.update(time.time() - start_time)

        # message = f'[ Round: {round_number} ' \
        #           f'| Local Train ' \
        #           f'| Client: {self.client_id} ' \
        #           f'| Epoch: {local_epoch}' \
        #           f'| Time: {batch_time.avg:.2f}s ' \
        #           f'| Loss: {losses.avg:.5f} ' \
        #           f'| Tr_Acc ({correct}/{data_size * local_epoch})={((correct / (data_size * local_epoch)) * 100):.2f}% ]'
        # logging.info(message)




        self.model.to("cpu")

    def train_foolsgold(self, round_number, local_epoch):
        # Set model to training mode and transfer to device.
        self.model.train()
        self.model.to(self.device)

        # load once to save time. To be used for calculating the accuracy.
        data_size = len(self.training_dataloader.dataset)
        # print("data_size",data_size) # 6000

        # Define attributes to keep track of scores during training
        batch_time = AverageMeter()
        losses = AverageMeter()
        start_time = time.time()
        correct = 0

        ##################
        last_local_model = dict()
        client_grad = []  # only works for aggr_epoch_interval=1
        epochs_local_update_list = []

        initial_global_model_params = self.parameters_to_vector(self.model.parameters()).detach()
        net=self.model
        for name, data in net.state_dict().items():
            last_local_model[name] = net.state_dict()[name].clone()

        epoch_loss = []

        ##################



        ##################more epoch
        for epoch in range(local_epoch):
            #################################
            target_params_variables = dict()
            for name, param in self.model.named_parameters():
                target_params_variables[name] = last_local_model[name].clone().detach().requires_grad_(False)

            batch_loss = []

            for batch_idx,(data, labels) in enumerate(self.training_dataloader):
                # Transfer data to CPU or GPU and set gradients to zero for performance.
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                self.optimizer.zero_grad()

                # Do a forward pass through the network to get prediction values and update loss metric.
                outputs = self.model(data)
                loss = self.loss_function(outputs, labels)

                # Do a backward pass through the network to get the gradients
                # and then use the optimizer to update the weights.
                loss.backward()


                ################
                # epochs_local_update_list
                for i, (name, params) in enumerate(self.model.named_parameters()):
                    if params.requires_grad:
                        if iter == 0 and batch_idx == 0:
                            client_grad.append(params.grad.clone())
                        else:
                            client_grad[i] += params.grad.clone()
                self.optimizer.step()
                batch_loss.append(loss.item())
                ################

                # Compare predictions to labels and get accuracy score.
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                # Update loss, accuracy and run_time metrics
                losses.update(loss.item())
                batch_time.update(time.time() - start_time)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

            local_model_update_dict = dict()
            for key, data in net.state_dict().items():
                # torch.zeros_like: 生成和括号内变量维度维度一致的全是零的内容。
                local_model_update_dict[key] = torch.zeros_like(data)
                local_model_update_dict[key] = (data - last_local_model[key])
                last_local_model[key] = copy.deepcopy(data)

            epochs_local_update_list.append(client_grad)
        # message = f'[ Round: {round_number} ' \
        #           f'| Local Train ' \
        #           f'| Client: {self.client_id} ' \
        #           f'| Epoch: {local_epoch}' \
        #           f'| Time: {batch_time.avg:.2f}s ' \
        #           f'| Loss: {losses.avg:.5f} ' \
        #           f'| Tr_Acc ({correct}/{data_size * local_epoch})={((correct / (data_size * local_epoch)) * 100):.2f}% ]'
        # logging.info(message)
        weight = np.float64(np.concatenate([param.data.cpu().numpy().flatten() for param in net.parameters()]))
        update = self.parameters_to_vector(net.parameters()).double() - initial_global_model_params

        return sum(epoch_loss) / len(epoch_loss), net.state_dict(), epochs_local_update_list, weight, update


        self.model.to("cpu")

    def parameters_to_vector(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
        r"""Convert parameters to one vector

        Args:
            parameters (Iterable[Tensor]): an iterator of Tensors that are the
                parameters of a model.

        Returns:
            The parameters represented by a single vector
        """
        # Flag for the device where the parameter is located
        param_device = None

        vec = []
        for param in parameters:
            # Ensure the parameters are located in the same device
            param_device = _check_param_device(param, param_device)

            vec.append(param.view(-1))
        return torch.cat(vec)

    def test(self, round_number):
        """
         Testing method for the client class.

         Parameters:
             round_number: number of the current epoch to be used for logging.
         """
        # Set model to testing mode and transfer to device.
        self.model.eval()
        self.model.to(self.device)

        # load once to save time. To be used for calculating the accuracy.
        data_size = len(self.training_dataloader.dataset)

        # Define attributes to keep track of scores during testing
        batch_time = AverageMeter()
        losses = AverageMeter()
        start_time = time.time()
        correct = 0

        with torch.no_grad():
            for data, labels in self.training_dataloader:
                # Transfer data to CPU or GPU.
                data, labels = data.float().to(self.device), labels.long().to(self.device)

                # Do a forward pass through the network to get prediction values and update loss metric.
                outputs = self.model(data)
                losses.update(self.loss_function(outputs, labels))

                # Compare predictions to labels and get accuracy score.
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                # Update time and accuracy metric.
                batch_time.update(time.time() - start_time)

            accuracy = (correct / data_size) * 100
            message = f'[ Round: {round_number} ' \
                      f'| Local Eval ' \
                      f'| Client: {self.client_id} ' \
                      f'| Loss: {losses.avg:.5f} ' \
                      f'| Test_Acc ({correct}/{data_size})={accuracy:.2f}% ] '
            logging.info(message)
            log_local = logging.getLogger('log_local')
            log_local.info(message)

        self.local_results = {"loss": [], "accuracy": []}
        self.local_results['loss'].append(losses.avg)
        self.local_results['accuracy'].append(accuracy)


