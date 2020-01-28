import torch
import torch.nn as nn
import torch.optim as torch_optim
import src.loss as custom_loss
import src.distilled.distilled_network as distilled_network
from pathlib import Path
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class LogitsTeacherFromFile(distilled_network.DistilledNet):
    def __init__(self,
                 teacher,
                 block,
                 num_blocks,
                 device=torch.device('cpu'),
                 use_hard_labels=False,
                 learning_rate=0.001,
                 scale_teacher_logits=False):

        super().__init__(teacher=teacher,
                         loss_function=custom_loss.gaussian_neg_log_likelihood,
                         device=device)

        self.use_hard_labels = use_hard_labels
        self.learning_rate = learning_rate
        self.scale_teacher_logits = scale_teacher_logits

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, 18)

        self.optimizer = torch_optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(self.device)


         # TODO: Temporary test of mse in beginning of training, SHOULD BE REMOVED BEFORE PUSHING (SORRY OTHERWISE)
        self.mse = False
        #self.loss = custom_loss.mse

        # Ad-hoc fix zero variance.

        self.variance_lower_bound = 0.001
        if self.variance_lower_bound > 0.0:
            self._log.warning("Non-zero variance lower bound set ({})".format(
                self.variance_lower_bound))

        self.optimizer = torch_optim.Adam(self.parameters(),
                                          lr=self.learning_rate)

        self.to(self.device)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def restore_loss(self):
        self.loss = custom_loss.gaussian_neg_log_likelihood
        self.mse = False

    def forward(self, x):
        """Estimate parameters of distribution
        """

        # x in this case will be a tuple
        #x = self.features(x[0])

        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        mid = int(x.shape[-1] / 2)
        mean = x[:, :mid]
        var_z = x[:, mid:]

        var = torch.log(1 + torch.exp(var_z)) + self.variance_lower_bound
        #var = torch.exp(var_z)

        return mean, var

    def _generate_teacher_predictions(self, inputs):
        """Generate teacher predictions"""

        logits = self.teacher.get_logits(inputs)

        if self.scale_teacher_logits:
            scaled_logits = logits - torch.stack([logits[:, :, -1]], axis=-1)
            logits = scaled_logits[:, :, :-1]

        return logits

    def predict(self, input_, num_samples=None):
        """Predict parameters
        Wrapper function for the forward function.
        """

        if isinstance(input_, list) or isinstance(input_, tuple):
            input_ = input_[0]

        if num_samples is None:
            num_samples = 50

        mean, var = self.forward(input_)

        samples = torch.zeros(
            [input_.size(0), num_samples,
             int(mean.size(-1))])
        for i in range(input_.size(0)):

            rv = torch.distributions.multivariate_normal.MultivariateNormal(
                loc=mean[i, :], covariance_matrix=torch.diag(var[i, :]))

            samples[i, :, :] = rv.rsample([num_samples])

        softmax_samples = torch.exp(samples) / (
            torch.sum(torch.exp(samples), dim=-1, keepdim=True) + 1)

        return softmax_samples

    def predict_logits(self, input_, num_samples=None):
        """Predict parameters
        Wrapper function for the forward function.
        """

        if isinstance(input_, list) or isinstance(input_, tuple):
            input_ = input_[0]

        if num_samples is None:
            num_samples = 50

        mean, var = self.forward(input_)

        samples = torch.zeros(
            [input_.size(0), num_samples,
             int(self.output_size / 2)])
        for i in range(input_.size(0)):

            rv = torch.distributions.multivariate_normal.MultivariateNormal(
                loc=mean[i, :], covariance_matrix=torch.diag(var[i, :]))

            samples[i, :, :] = rv.rsample([num_samples])

        return samples

    def _learning_rate_condition(self, epoch=None):
        """Evaluate condition for increasing learning rate
        Defaults to never increasing. I.e. returns False
        """

        return True

    def calculate_loss(self, outputs, teacher_predictions, labels=None):
        """Calculate loss function
        Wrapper function for the loss function.
        """

        if self.mse:
            outputs = outputs[0]

        return self.loss(outputs, teacher_predictions)

    # To use as a metric
    def mean_expected_value(self, outputs, teacher_predictions):
        exp_value = outputs[0]

        return torch.mean(exp_value, dim=0)

    # To use as a metric
    def mean_variance(self, outputs, teacher_predictions):
        variance = outputs[1]

        return torch.mean(variance, dim=0)


    def eval_mode(self, train=False):
        # Setting layers to eval mode

        if train:
            # Not sure this is the way to go
            self.conv1.train()
            self.bn1.train()
            self.layer1.train()
            self.layer2.train()
            self.layer3.train()
            self.layer4.train()
            self.linear.train()
        else:
            self.conv1.eval()
            self.bn1.eval()
            self.layer1.eval()
            self.layer2.eval()
            self.layer3.eval()
            self.layer4.eval()
            self.linear.eval()

    # This is for the gpu
    def save_weights(self):

        filepath = "distilled_model_"

        layers = [self.conv1, self.bn1]

        block_list = [self.layer1, self.layer2, self.layer3, self.layer4]

        for block in list(block_list): # TODO: check this
            for layer in list(block.modules()):
                layers.append(layer)

        layers.append(self.linear)

        counter_1 = 0
        counter_2 = 0
        for layer in layers:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                torch.save(layer.weight.data, Path(filepath + "weight" + str(counter_1)))
                torch.save(layer.bias.data, Path(filepath + "bias" + str(counter_1)))
                counter_1 += 1

            elif isinstance(layer, nn.BatchNorm2d):
                torch.save(layer.weight.data, Path(filepath + "bn_weight" + str(counter_2)))
                torch.save(layer.bias.data, Path(filepath + "bn_bias" + str(counter_2)))
                torch.save(layer.running_mean.data, Path(filepath + "bn_running_mean" + str(counter_2)))
                torch.save(layer.running_var.data, Path(filepath + "bn_running_var" + str(counter_2)))
                counter_2 += 1

    def load_weights(self, file_dir=None):

        if file_dir is None:
            filepath = "distilled_model_"
        else:
            filepath = file_dir + "distilled_model_"


        # TODO: CHECK SO THAT THE LOADING WORKS
        layers = [self.conv1, self.bn1]

        block_list = [self.layer1, self.layer2, self.layer3, self.layer4]

        for block in list(block_list):  # TODO: check this
            for layer in list(block.modules()):
                layers.append(layer)

        layers.append(self.linear)

        counter_1 = 0
        counter_2 = 0

        for layer in layers:
            # Set layer to evaluation mode
            layer.eval()

            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):

                loaded_weight = torch.load(Path(filepath + "weight" + str(counter_1)), map_location=self.device)
                assert layer.weight.data.shape == loaded_weight.shape
                layer.weight.data = loaded_weight

                loaded_bias = torch.load(Path(filepath + "bias" + str(counter_1)), map_location=self.device)
                assert layer.bias.data.shape == loaded_bias.shape
                layer.bias.data = loaded_bias

                counter_1 += 1

            elif isinstance(layer, nn.BatchNorm2d):

                loaded_weight = torch.load(Path(filepath + "bn_weight" + str(counter_2)), map_location=self.device)
                assert layer.weight.data.shape == loaded_weight.shape
                layer.weight.data = loaded_weight

                loaded_bias = torch.load(Path(filepath + "bn_bias" + str(counter_2)), map_location=self.device)
                assert layer.bias.data.shape == loaded_bias.shape
                layer.bias.data = loaded_bias

                # RÅKADE SPARA PARAMETERN OCH INTE SJÄLVA DATATENSORN HÄR, MEN SKA FIXA DET SEN
                loaded_running_mean = torch.load(Path(filepath + "bn_running_mean" + str(counter_2)),
                                                 map_location=self.device)
                assert layer.running_mean.shape == loaded_running_mean.shape
                layer.running_mean = loaded_running_mean

                loaded_running_var = torch.load(Path(filepath + "bn_running_var" + str(counter_2)),
                                                map_location=self.device)
                assert layer.running_var.shape == loaded_running_var.shape
                layer.running_var = loaded_running_var

                counter_2 += 1




