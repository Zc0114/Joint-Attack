import torch
import torch.nn as nn
import torch.functional as F
""" Ja hier moet dus documentatie """


########################################################################################################################
# Model Class Helpers
########################################################################################################################


def init_weights_normal(layer):
    """
     Applies L2 regularization to a given model layer
        Set to use normal distribution
     """
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, 0.0, 0.01)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def get_output_shape_of_last_layer(target_model):
    """
        Gets the output shape of the last layer of a given model
    """
    output_shape = None
    #####always select last layer weight shape
    for name, param in target_model.named_parameters():
        if 'weight' in name:
            output_shape = param.size()[0]
    return output_shape


def get_last_layer_name(target_model):
    """
         Gets the name of the last layer of a given model
    """
    layer_name = None
    for name, param in target_model.named_parameters():
        if 'weight' in name:
            layer_name = name
    return layer_name


########################################################################################################################
# Models
########################################################################################################################


class TestNet(nn.Module):

    def __init__(self, input_size=3 * 32 * 32, output_classes=100):
        super(TestNet, self).__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_classes)
        )

    def forward(self, x):
        return self.net(x)


class AlexNet(nn.Module):

    def __init__(self, num_classes=100):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x






#############################最初的
class AttackModel(nn.Module):
    """ Ja hier moet dus documentatie """

    def __init__(self, target_model,
                 number_of_observed_models=1,
                 number_of_classes=100,
                 linearsize = 256,
                 kernel = 100
                 ):
        super(AttackModel, self).__init__()
        self.last_layer_name = get_last_layer_name(target_model)
        self.encoder_inputs = []
        self.model = target_model
        self.number_of_observed_models = number_of_observed_models

        # create component if last layer is to be exploited
        ####input number_of_classes-100-64 liner
        self.layer_component = FcnComponent(number_of_classes, 100)
        module_output = get_output_shape_of_last_layer(self.layer_component)
        self.encoder_inputs.append(module_output)

        # create component if OHE label is to be exploited
        ###input number_of_classes-128-64 liner
        self.label_component = FcnComponent(number_of_classes, 128)
        module_output = get_output_shape_of_last_layer(self.label_component)
        self.encoder_inputs.append(module_output)

        # create component if loss value is to be exploited
        ###input 1-10-64 liner
        self.loss_component = FcnComponent(1, number_of_classes)
        module_output = get_output_shape_of_last_layer(self.loss_component)
        self.encoder_inputs.append(module_output)


        # # create component if loss value is to be exploited
        # ###input 1-10-64 liner
        # self.loss_diff_component = FcnComponent(1, number_of_classes)
        # module_output = get_output_shape_of_last_layer(self.loss_diff_component)
        # self.encoder_inputs.append(module_output)

        # creates CNN/FCN component for gradient values of layers of gradients to exploit
        ###input

        self.gradient_component = GradientComponent(number_of_classes=number_of_classes,linearsize=linearsize,kernel=kernel)
        module_output = get_output_shape_of_last_layer(self.gradient_component)
        self.encoder_inputs.append(module_output)


        # creates CNN/FCN component for gradient values of layers of gradients to exploit
        ###input
        # self.gradient_diff_component = GradientComponent(number_of_classes=number_of_classes,linearsize=linearsize,kernel=kernel)
        # module_output = get_output_shape_of_last_layer(self.gradient_diff_component)
        # self.encoder_inputs.append(module_output)


        # self.ImgComponent=ImgComponent()

        ###
        self.encoder = AutoEncoder(self.encoder_inputs, number_of_observed_models)
        self.output = nn.Sigmoid()

    def forward(self, model_predictions, encoded_labels, loss_values, gradients) -> torch.Tensor:

        for model_input in range(self.number_of_observed_models):
            temp = self.layer_component(model_predictions[model_input])###64*100-64*64  feature 100-64
            temp1 = self.label_component(encoded_labels)###64*100-64*64  feature 100-64
            temp2 = self.loss_component(loss_values[model_input])#64*1-64*64 feature 1-64
            temp3 = self.gradient_component(gradients[model_input])#64*1*256*100-
            #temp4=self.loss_diff_component(loss_diff[model_input])
            #temp5=self.gradient_diff_component(grad_diff[model_input])
            #temp6=self.ImgComponent(img[model_input])
            if model_input == 0:
                x = torch.cat((temp, temp1, temp2, temp3), dim=1)
            else:
                x = torch.cat((temp, temp1, temp2, temp3, x), dim=1)

        is_member = self.encoder(x)
        return self.output(is_member)











##############to do
class AttackModel(nn.Module):
    """ Ja hier moet dus documentatie """

    def __init__(self, target_model,
                 number_of_observed_models=1,
                 number_of_classes=100,
                 linearsize = 256,
                 kernel = 100
                 ):
        super(AttackModel, self).__init__()
        self.last_layer_name = get_last_layer_name(target_model)
        self.encoder_inputs = []
        self.model = target_model
        self.number_of_observed_models = number_of_observed_models

        # create component if last layer is to be exploited
        ####input number_of_classes-100-64 liner
        self.layer_component = FcnComponent(number_of_classes, 100)
        module_output = get_output_shape_of_last_layer(self.layer_component)
        self.encoder_inputs.append(module_output)

        # create component if OHE label is to be exploited
        ###input number_of_classes-128-64 liner
        self.label_component = FcnComponent(number_of_classes, 128)
        module_output = get_output_shape_of_last_layer(self.label_component)
        self.encoder_inputs.append(module_output)

        # create component if loss value is to be exploited
        ###input 1-10-64 liner
        self.loss_component = FcnComponent(1, number_of_classes)
        module_output = get_output_shape_of_last_layer(self.loss_component)
        self.encoder_inputs.append(module_output)


        # # create component if loss value is to be exploited
        # ###input 1-10-64 liner
        # self.loss_diff_component = FcnComponent(1, number_of_classes)
        # module_output = get_output_shape_of_last_layer(self.loss_diff_component)
        # self.encoder_inputs.append(module_output)

        # creates CNN/FCN component for gradient values of layers of gradients to exploit
        ###input

        self.gradient_component = GradientComponent(number_of_classes=number_of_classes,linearsize=linearsize,kernel=kernel)
        module_output = get_output_shape_of_last_layer(self.gradient_component)
        self.encoder_inputs.append(module_output)


        # creates CNN/FCN component for gradient values of layers of gradients to exploit
        ###input
        # self.gradient_diff_component = GradientComponent(number_of_classes=number_of_classes,linearsize=linearsize,kernel=kernel)
        # module_output = get_output_shape_of_last_layer(self.gradient_diff_component)
        # self.encoder_inputs.append(module_output)


        #self.ImgComponent=ImgComponent()

        ###
        self.encoder = AutoEncoder(self.encoder_inputs, number_of_observed_models)
        self.output = nn.Sigmoid()

    def forward(self, model_predictions, encoded_labels, loss_values, gradients) -> torch.Tensor:

        for model_input in range(self.number_of_observed_models):
            temp = self.layer_component(model_predictions[model_input])###64*100-64*64  feature 100-64
            temp1 = self.label_component(encoded_labels)###64*100-64*64  feature 100-64
            temp2 = self.loss_component(loss_values[model_input])#64*1-64*64 feature 1-64
            temp3 = self.gradient_component(gradients[model_input])#64*1*256*100-
            #temp6=self.ImgComponent(img[model_input])
            if model_input == 0:
                x = torch.cat((temp, temp1, temp2, temp3), dim=1)
            else:
                ### where x??
                x = torch.cat((temp, temp1, temp2, temp3, x), dim=1)

        is_member = self.encoder(x)
        return self.output(is_member)


class AttackModel2(nn.Module):
    """ Ja hier moet dus documentatie """

    def __init__(self, target_model,
                 number_of_observed_models=1,
                 number_of_classes=100,
                 linearsize = 256,
                 kernel = 100,
                 ismnist=1
                 ):
        super(AttackModel2, self).__init__()
        self.last_layer_name = get_last_layer_name(target_model)
        self.encoder_inputs = []
        self.model = target_model
        self.number_of_observed_models = number_of_observed_models

        # create component if last layer is to be exploited
        ####input number_of_classes-100-64 liner
        self.layer_component = FcnComponent(number_of_classes, 100)
        module_output = get_output_shape_of_last_layer(self.layer_component)
        self.encoder_inputs.append(module_output)

        # create component if OHE label is to be exploited
        ###input number_of_classes-128-64 liner
        self.label_component = FcnComponent(number_of_classes, 128)
        module_output = get_output_shape_of_last_layer(self.label_component)
        self.encoder_inputs.append(module_output)

        # create component if loss value is to be exploited
        ###input 1-10-64 liner
        self.loss_component = FcnComponent(1, number_of_classes)
        module_output = get_output_shape_of_last_layer(self.loss_component)
        self.encoder_inputs.append(module_output)


        # create component if loss value is to be exploited
        ###input 1-10-64 liner
        self.loss_diff_component = FcnComponent(1, number_of_classes)
        module_output = get_output_shape_of_last_layer(self.loss_diff_component)
        self.encoder_inputs.append(module_output)

        # creates CNN/FCN component for gradient values of layers of gradients to exploit
        ###input

        self.gradient_component = GradientComponent(number_of_classes=number_of_classes,linearsize=linearsize,kernel=kernel)
        module_output = get_output_shape_of_last_layer(self.gradient_component)
        self.encoder_inputs.append(module_output)


        # creates CNN/FCN component for gradient values of layers of gradients to exploit
        ###input
        self.gradient_diff_component = GradientComponent(number_of_classes=number_of_classes,linearsize=linearsize,kernel=kernel)
        module_output = get_output_shape_of_last_layer(self.gradient_diff_component)
        self.encoder_inputs.append(module_output)


        self.ImgComponent=ImgComponent(ismnist)
        module_output = get_output_shape_of_last_layer(self.ImgComponent)
        self.encoder_inputs.append(module_output)
        ###
        self.encoder = AutoEncoder(self.encoder_inputs, number_of_observed_models)
        self.output = nn.Sigmoid()

    def forward(self, model_predictions, encoded_labels, loss_values, gradients,loss_diff,grad_diff,img) -> torch.Tensor:

        for model_input in range(self.number_of_observed_models):
            temp = self.layer_component(model_predictions[model_input])###64*100-64*64  feature 100-64
            temp1 = self.label_component(encoded_labels)###64*100-64*64  feature 100-64
            temp2 = self.loss_component(loss_values[model_input])#64*1-64*64 feature 1-64
            temp3 = self.gradient_component(gradients[model_input])#64*1*256*100-
            temp4=self.loss_diff_component(loss_diff[model_input])
            temp5=self.gradient_diff_component(grad_diff[model_input])
            temp6=self.ImgComponent(img[model_input])
            if model_input == 0:
                x = torch.cat((temp, temp1, temp2, temp3,temp4,temp5,temp6), dim=1)
            else:
                ### where x??
                x = torch.cat((temp, temp1, temp2, temp3,temp4,temp5,temp6, x), dim=1)

        is_member = self.encoder(x)
        return self.output(is_member)


class AutoEncoder(nn.Module):
    """ Ja hier moet dus documentatie """

    def __init__(self, encoder_input, number_of_observed_models):
        """ Ja hier moet dus documentatie """
        super(AutoEncoder, self).__init__()
        input_concat = sum(encoder_input)###???? why sum?

        self.encoder = nn.Sequential(
            nn.Linear(input_concat * number_of_observed_models, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """ Ja hier moet dus documentatie """
        return self.encoder(x)


class FcnComponent(nn.Module):

    def __init__(self, input_size, layer_size=128):
        """ Ja hier moet dus documentatie """
        super(FcnComponent, self).__init__()

        self.fcn = nn.Sequential(
            nn.Linear(input_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, 64),
            nn.ReLU()
        )
        self.fcn.apply(init_weights_normal)

    def forward(self, x):
        """ Ja hier moet dus documentatie """
        return self.fcn(x)


class GradientComponent(nn.Module):

    def __init__(self, number_of_classes=100,linearsize=256,kernel=100):
        """ Ja hier moet dus documentatie """
        super(GradientComponent, self).__init__()

        self.cnn = nn.Sequential(
            nn.Dropout(p=0.2),
            #nn.Conv2d(1, 1000, kernel_size=(1, 100), stride=
            nn.Conv2d(1, 500, kernel_size=(1, kernel), stride=1),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(linearsize * 500, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.linear.apply(init_weights_normal)

    def forward(self, x):
        """ Ja hier moet dus documentatie """
        temp = self.cnn(x).view(x.size()[0], -1)
        return self.linear(temp)



class ImgComponent(nn.Module):

    def __init__(self,ismnist):
        """ Ja hier moet dus documentatie """
        super(ImgComponent, self).__init__()
        self.ismnist=ismnist
        if self.ismnist==1:
            self.cnn = nn.Sequential(
                #nn.Conv2d(1, 1000, kernel_size=(1, 100), stride=
                #nn.Conv2d(3, 500, kernel_size=(3, 3), stride=1),
                nn.Conv2d(1, 500, kernel_size=(3, 3), stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(p=0.2)
            )
            self.linear = nn.Sequential(
                nn.Dropout(p=0.2),
                #nn.Linear(15*15*500, 1024),
                nn.Linear(13*13*500, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                #nn.Conv2d(1, 1000, kernel_size=(1, 100), stride=
                nn.Conv2d(3, 500, kernel_size=(3, 3), stride=1),
                #nn.Conv2d(1, 500, kernel_size=(3, 3), stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(p=0.2)
            )
            self.linear = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(15*15*500, 1024),
                #nn.Linear(13*13*500, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
            )
        
        self.linear.apply(init_weights_normal)

    def forward(self, x):
        """ Ja hier moet dus documentatie """
        temp = self.cnn(x).view(x.size()[0], -1)
        return self.linear(temp)



class CNNCifar10(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNCifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)#last layer name
        self.relu=nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5(nn.Module):
    def __init__(self,num_class):
        super().__init__()
        #####################
        #LeNet5 in_channels need change if input chanle is 3
        #####################
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_class)
        self.relu=nn.ReLU()

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = self.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = self.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 5 * 5 * 16)
        tensor = self.relu(self.fc1(tensor))
        tensor = self.relu(self.fc2(tensor))
        tensor = self.fc3(tensor)#last layer
        return tensor



class CNNMnist(nn.Module):
    def __init__(self,num_classes=10):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(0.1)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50,num_classes)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.1)
        self.max_pool2d=nn.MaxPool2d(2)

    def forward(self, x):
        x=self.conv1(x)
        x=self.max_pool2d(x)
        x = self.relu(x)
        x=self.conv2(x)
        x=self.conv2_drop(x)
        x=self.max_pool2d(x)
        x = self.relu(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x=self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CNNFashion_Mnist(nn.Module):
    def __init__(self,num_class=10):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, num_class)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out





class CNNModel(nn.Module):
	def __init__(self):
		super (CNNModel, self).__init__()

		self.cnn1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5, stride = 1, padding = 2)
		self.relu1 = nn.ReLU()
		self.norm1 = nn.BatchNorm2d(32)
		nn.init.xavier_uniform(self.cnn1.weight)

		self.maxpool1 = nn.MaxPool2d(kernel_size=2)

		self.cnn2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 2)
		self.relu2 = nn.ReLU()
		self.norm2 = nn.BatchNorm2d(64)
		nn.init.xavier_uniform(self.cnn2.weight)

		self.maxpool2 = nn.MaxPool2d(kernel_size=2)

		self.fc1 = nn.Linear(4096, 256)
		self.fcrelu = nn.ReLU()

		self.fc2 = nn.Linear(256, 10)

	def forward(self, x):
		out = self.cnn1(x)
		out = self.relu1(out)
		out = self.norm1(out)

		out = self.maxpool1(out)

		out = self.cnn2(out)
		out = self.relu2(out)
		out = self.norm2(out)

		out = self.maxpool2(out)

		out = out.view(out.size(0),-1)

		out = self.fc1(out)
		out = self.fcrelu(out)

		out = self.fc2(out)
		return out



class CNNNet(nn.Module):
    def __init__(self, num_classes=100):
        super(CNNNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x




