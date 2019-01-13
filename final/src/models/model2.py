import torch
from torchvision import models
from pretrainedmodels.models import bninception
from torch import nn
from config import config
from collections import OrderedDict
import torch.nn.functional as F

def get_net():
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()

            self.conv1   = nn.Conv2d(4, 8, kernel_size=3)
            self.conv2   = nn.Conv2d(8, 8, kernel_size=3)
            self.conv3   = nn.Conv2d(8, 16, kernel_size=3)
            self.conv4_1 = nn.Conv2d(16, 16, padding=0,kernel_size=3)
            self.conv4_2 = nn.Conv2d(16, 16, padding=1,kernel_size=5)
            self.conv4_3 = nn.Conv2d(16, 16, padding=2,kernel_size=7)
            self.conv4_4 = nn.Conv2d(16, 16, padding=3,kernel_size=9)
            self.conv5   = nn.Conv2d(16, 32, kernel_size=3)
            self.conv6   = nn.Conv2d(32, 64, kernel_size=3)
            self.conv7   = nn.Conv2d(64, 128, kernel_size=3)
            self.global_pool = nn.AvgPool2d (7, stride=1, padding=0, ceil_mode=True, count_include_pad=True)
            self.dense = nn.Linear(2768896, 128)
            #self.last_linear = nn.Linear (1024, num_classes)
            self.last_linear = nn.Sequential(
                        nn.BatchNorm1d(1024),
                        nn.Dropout(0.5),
                        nn.Linear(1024, config.num_classes),
                    )
            self.output = nn.Linear(128, config.num_classes)


        def features(self, x):
            y = self.conv1(x)
            y = self.conv2(y)
            y = F.max_pool2d(self.conv3(y), 2)

            y_1 = self.conv4_1(y)
            y_2 = self.conv4_2(y)
            y_3 = self.conv4_3(y)
            y_4 = self.conv4_4(y)

            y = torch.cat([y_1, y_2, y_3, y_4], dim=0)

            y = F.max_pool2d(y, 2)
            y = F.max_pool2d(self.conv5(y), 2)
            y = F.max_pool2d(self.conv6(y), 2)
            y = F.max_pool2d(self.conv7(y), 2)
            
            y = y.view(y.size(0), -1)
            y = self.global_pool(y)
            y = y.view(1, -1)
            y = self.dense(y)
            y = self.output(y)
            y = F.softmax(self.output(y))

            return y
        def forward(self, input):
            x = self.features(input)
            x = self.logits(x)
            return x
    return Model()
