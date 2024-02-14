import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable

class simplenet(nn.Module):
    def __init__(self, classes=10, simpnet_name='simplenet'):
        super(simplenet, self).__init__()
        #print(simpnet_name)
        self.features = self._make_layers() #self._make_layers(cfg[simpnet_name])
        self.classifier = nn.Linear(256, classes)
        self.drp = nn.Dropout(0.1)

    def load_my_state_dict(self, state_dict):

        own_state = self.state_dict()

        # print(own_state.keys())
        # for name, val in own_state:
        # print(name)
        for name, param in state_dict.items():
            name = name.replace('module.', '')
            if name not in own_state:
                # print(name)
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            print("STATE_DICT: {}".format(name))
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ... Using Initial Params'.format(
                    name, own_state[name].size(), param.size()))

    def forward(self, x):
        out = self.features(x)

        #Global Max Pooling
        out = F.max_pool2d(out, kernel_size=out.size()[2:]) 
        # out = F.dropout2d(out, 0.1, training=True)
        out = self.drp(out)

        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self):

        model = nn.Sequential(
                             nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(64, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),

                             nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),

                             nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),

                             nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),


                             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                             nn.Dropout2d(p=0.1),


                             nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),

                             nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),

                             nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),



                             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                             nn.Dropout2d(p=0.1),


                             nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),


                             nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),



                             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                             nn.Dropout2d(p=0.1),



                             nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(512, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),



                             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                             nn.Dropout2d(p=0.1),


                             nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
                             nn.BatchNorm2d(2048, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),



                             nn.Conv2d(2048, 256, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
                             nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),


                             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                             nn.Dropout2d(p=0.1),


                             nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),

                            )

        for m in model.modules():
          if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

        return model

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())
        # self.fc1 = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        # out = self.fc1(out)
        out = self.fc2(out)
        return out
