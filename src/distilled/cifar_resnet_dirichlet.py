import torch
import torch.nn as nn
import torch.optim as torch_optim
import src.loss as custom_loss
import src.distilled.distilled_network as distilled_network
import torch.nn.functional as F


class CifarResnetDirichlet(distilled_network.DistilledNet):
    def __init__(self,
                 teacher,
                 block,
                 num_blocks,
                 device=torch.device('cpu'),
                 use_hard_labels=False,
                 learning_rate=0.001,
                 vanilla_distillation=False):

        super().__init__(teacher=teacher,
                         loss_function=custom_loss.dirichlet_neg_log_likelihood,
                         device=device)

        self.use_hard_labels = use_hard_labels
        self.learning_rate = learning_rate
        self.vanilla_distillation = vanilla_distillation

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.output_size = 10
        self.linear = nn.Linear(512 * block.expansion, self.output_size)

        self.optimizer = torch_optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(self.device)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, softmax_transform=True):
        """Estimate parameters of distribution
        """

        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = torch.exp(self.linear(out))

        return out

    def _generate_teacher_predictions(self, inputs, t=1, gamma=1e-4):
        """Generate teacher predictions"""

        # Minska t här (ha som typ en egenskap)?
        logits = self.teacher.get_predictions(inputs)
        predictions = torch.exp(logits / t) / torch.sum(torch.exp(logits / t), dim=-1, keepdim=True)

        # "Central smoothing"
        predictions = (1 - gamma) * predictions + gamma * (1 / self.output_size)

        return predictions

    def predict(self, input_, num_samples=None, return_params=False):
        """Predict parameters
        Wrapper function for the forward function.
        """

        if isinstance(input_, list) or isinstance(input_, tuple):
            input_ = input_[0]

        if num_samples is None:
            num_samples = 100

        alphas = self.forward(input_)

        samples = torch.zeros(
            [input_.size(0), num_samples, self.output_size])

        # TODO: I think we can do this:
        # rv = torch.distributions.dirichlet.Dirichlet(concentration=alphas)
        # samples = rv.rsample([num_samples]) + maybe some tranpose
        for i in range(input_.size(0)):

            rv = torch.distributions.dirichlet.Dirichlet(concentration=alphas)
            samples[i, :, :] = rv.rsample([num_samples])

        if return_params:
            return samples, alphas

        else:
            return samples

    def _learning_rate_condition(self, epoch):
        step_epochs = [80, 120, 160, 180]
        if epoch in step_epochs:
            return True
        else:
            return False

    def calculate_loss(self, outputs, teacher_predictions, labels=None):
        """Calculate loss function
        Wrapper function for the loss function.
        """

        return self.loss(outputs, teacher_predictions)

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






