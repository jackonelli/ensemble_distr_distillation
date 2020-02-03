import torch
import torch.optim as torch_optim
import torch.nn as nn
from src.ensemble import ensemble


class Resnet20(ensemble.EnsembleMember):
    def __init__(self,
                 features_list,
                 device=torch.device("cpu"),
                 learning_rate=0.001):

        super().__init__(output_size=10, loss_function=nn.NLLLoss(), device=device)
        self.learning_rate = learning_rate

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

        self.optimizer = torch_optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(self.device)

    def forward(self, x):

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

        return x

    def transform_logits(self, logits):
        return logits

    def calculate_loss(self, outputs, labels):
        return self.loss(outputs, labels.type(torch.LongTensor))

    def predict(self, x, t=1):
        x = self.forward(x)
        x = (nn.Softmax(dim=-1))(x)

        return x

    def _learning_rate_condition(self, epoch):
        step_epochs = [80, 120, 160, 180]
        if epoch in step_epochs:
            return True
        else:
            return False
