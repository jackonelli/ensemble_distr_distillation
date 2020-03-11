import torch
import torch.optim as torch_optim
import torch.nn as nn
from src.ensemble import ensemble


class SimpleClassifier(ensemble.EnsembleMember):
    def __init__(self,
                 layer_sizes,
                 device=torch.device("cpu"),
                 learning_rate=0.001):

        super().__init__(loss_function=nn.NLLLoss(), device=device)
        self.learning_rate = learning_rate

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        self.optimizer = torch_optim.SGD(self.parameters(),
                                         lr=self.learning_rate,
                                         momentum=0.9)
        self.to(self.device)

    def forward(self, x):

        for layer in self.layers[:-1]:
            x = nn.functional.relu(layer(x))

        x = self.layers[-1](x)

        return x

    def transform_logits(self, logits):
        return (nn.Softmax(dim=-1))(logits)

    def calculate_loss(self, outputs, labels):
        log_outputs = torch.log(outputs)
        # Removing this since it gives nan loss

        return self.loss(log_outputs, labels.type(torch.LongTensor))

    def predict(self, x, t=1):
        x = self.forward(x)
        x = self.transform_logits(x)

        return x
