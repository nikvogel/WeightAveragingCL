import avalanche.models
from avalanche.models import MultiHeadClassifier, MultiTaskModule, BaseModel
from torch import nn
import torch
import torchvision

class MultiHeadMLP(MultiTaskModule):
    def __init__(self, input_size=28 * 28, hidden_size=256, hidden_layers=2,
                 drop_rate=0, relu_act=True):
        super().__init__()
        self._input_size = input_size

        layers = nn.Sequential(*(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
                                 nn.Dropout(p=drop_rate)))
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}", nn.Sequential(
                    *(nn.Linear(hidden_size, hidden_size),
                      nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
                      nn.Dropout(p=drop_rate))))

        self.features = nn.Sequential(*layers)
        self.classifier = MultiHeadClassifier(hidden_size)

    def forward(self, x, task_labels):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x, task_labels)
        return x


class MLP(nn.Module, BaseModel):
    def __init__(self, input_size=28 * 28, hidden_size=256, hidden_layers=2,
                 output_size=10, drop_rate=0, relu_act=True, initial_out_features=0):
        """
        :param initial_out_features: if >0 override output size and build an
            IncrementalClassifier with `initial_out_features` units as first.
        """
        super().__init__()
        self._input_size = input_size

        layers = nn.Sequential(*(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
                                 nn.Dropout(p=drop_rate)))
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}", nn.Sequential(
                    *(nn.Linear(hidden_size, hidden_size),
                      nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
                      nn.Dropout(p=drop_rate))))

        self.features = nn.Sequential(*layers)

        if initial_out_features > 0:
            self.classifier = avalanche.models.IncrementalClassifier(in_features=hidden_size,
                                                                     initial_out_features=initial_out_features)
        else:
            self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        return self.features(x)


class SI_CNN(MultiTaskModule):
    def __init__(self, hidden_size=512):
        super().__init__()
        layers = nn.Sequential(*(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=(1, 1)),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d((2, 2)),
                                 nn.Dropout(p=0.25),
                                 nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d((2, 2)),
                                 nn.Dropout(p=0.25),
                                 nn.Flatten(),
                                 nn.Linear(2304, hidden_size),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=0.5)
                                 ))
        self.features = nn.Sequential(*layers)
        self.classifier = MultiHeadClassifier(hidden_size, initial_out_features=10)

    def forward(self, x, task_labels):
        x = self.features(x)
        x = self.classifier(x, task_labels)
        return x


class FlattenP(nn.Module):
    '''A nn-module to flatten a multi-dimensional tensor to 2-dim tensor.'''

    def forward(self, x):
        batch_size = x.size(0)   # first dimenstion should be batch-dimension.
        return x.view(batch_size, -1)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr


class MLP_gss(nn.Module):
    def __init__(self, sizes, bias=True):
        super(MLP_gss, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            if i < (len(sizes)-2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))

        self.net = nn.Sequential(FlattenP(), *layers)

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, N, num_classes):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, N, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(N),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(N, 2*N, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2*N),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(2*N, 4*N, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*N),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.classifier = nn.Sequential(nn.Linear(4*N, num_classes))
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CNN_5(nn.Module):
    def __init__(self, N, num_classes=200):
        super(CNN_5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, N, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(N),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(N, 2*N, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(2*N),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer3 = nn.Sequential(
            nn.Conv2d(2*N, 4*N, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(4*N),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer4 = nn.Sequential(
            nn.Conv2d(4*N, 8*N, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8*N),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer5 = nn.Sequential(
            nn.Conv2d(8*N, 16*N, kernel_size=2, stride=2, padding=2),
            nn.BatchNorm2d(16*N),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.classifier = nn.Sequential(nn.Linear(16*N, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class CNN_6(nn.Module):
    def __init__(self, N, num_classes):
        super(CNN_6, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, N, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(N),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(N, 2*N, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2*N),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(2*N, 4*N, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*N),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(4*N, 8*N, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8*N),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(8*N, 16*N, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16*N),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer6 = nn.Sequential(
            nn.Conv2d(16*N, 32*N, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32*N),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.classifier = nn.Sequential(nn.Linear(32*N, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CNN_Single(nn.Module):
    def __init__(self, N, num_classes):
        super(CNN_Single, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, N, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(N),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(N, 2*N, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2*N),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(2*N, 4*N, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4*N),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.classifier = nn.Sequential(nn.Linear(4*N, num_classes))
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



__all__ = ['MultiHeadMLP', 'MLP', 'SI_CNN', 'MLP_gss', 'CNN', 'CNN_6','CNN_Single', 'CNN_5']
