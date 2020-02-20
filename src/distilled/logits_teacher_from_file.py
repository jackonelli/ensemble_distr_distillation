import torch
import torch.nn as nn
import torch.optim as torch_optim
import src.loss as custom_loss
import src.distilled.distilled_network as distilled_network
from pathlib import Path


# TODO: Remove if we don't use for other purpose than resnet + cifar10
class LogitsTeacherFromFile(distilled_network.DistilledNet):
    def __init__(self,
                 features_list,
                 teacher,
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

        # I do this ugly thing here since I have no better idea right now
        self.f1 = features_list[0]
        self.f2 = features_list[1]
        self.f3 = features_list[2]
        self.f4 = features_list[3]
        self.f5 = features_list[4]
        self.f6 = features_list[5]
        self.f7 = features_list[6]
        self.f8 = features_list[7]
        self.f9 = features_list[8]
        self.f10 = features_list[9]
        self.f11 = features_list[10]
        self.f12 = features_list[11]
        self.f13 = features_list[12]
        self.f14 = features_list[13]
        self.f15 = features_list[14]
        self.f16 = features_list[15]
        self.f17 = features_list[16]
        #self.features = features_list

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

        # BLOCK 1
        x = self.f1(x)
        y = self.f2(x)
        x = nn.ReLU()(x + y)
        y = self.f3(x)
        x = nn.ReLU()(x + y)
        y = self.f4(x)
        x = nn.ReLU()(x + y)

        # BLOCK 2
        y = self.f5(x)
        x = self.f6(x)
        x = nn.ReLU()(x + y)
        y = self.f7(x)
        x = self.f8(x)
        x = nn.ReLU()(x + y)
        y = self.f9(x)
        x = self.f10(x)
        x = nn.ReLU()(x + y)

        # BLOCK 3
        y = self.f11(x)
        x = self.f12(x)
        x = nn.ReLU()(x + y)
        y = self.f13(x)
        x = self.f14(x)
        x = nn.ReLU()(x + y)
        y = self.f15(x)
        x = self.f16(x)
        x = nn.ReLU()(x + y)
        x = self.f17(x)

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

        if self.scale_teacher_logits:
            samples = torch.cat((samples, torch.zeros(samples.size(0), num_samples, 1)))

        return nn.Softmax(dim=-1)(samples)

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

        if self.scale_teacher_logits:
            samples = torch.cat((samples, torch.zeros(samples.size(0), num_samples, 1)), dim=-1)

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


