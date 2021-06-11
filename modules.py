import numpy as np
from torch import nn
from utils import weight_init_normal

class FMnistMolel(nn.Module):
    def __init__(self, id=1, T=2.0, channel=1):
        super().__init__()
        if id not in [1, 2, 3, 4]:
            raise ValueError('Invalid Model type')
            sys.exit()
        self.id = id
        self.channel = channel
        self.T =T
        # id = 1:mtl method
        # id = 2:autoencoder id
        # id = 3:Output all
        # id = 4:cls only for attack
        self.share_layer = nn.Sequential(
            nn.Conv2d(self.channel, 16, 5, padding=2), nn.Tanh(), nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 5, padding=2), nn.Tanh(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 96, 5, padding=2), nn.Tanh(), nn.BatchNorm2d(96),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 160, 5, padding=2), nn.Tanh(), nn.BatchNorm2d(160),
            nn.MaxPool2d(2),
            nn.Conv2d(160, 256, 3, padding=1), nn.Tanh(), nn.BatchNorm2d(256),
        )
        self.cls1 = nn.Sequential(
            nn.Conv2d(256, 10, 4)
        )
        self.cls2 = nn.LogSoftmax(dim=1)
        self.distr = nn.Softmax(dim=1)

        self.AEmtl = nn.Sequential(
            nn.Conv2d(256, 160, 5, padding=2), nn.Tanh(), nn.BatchNorm2d(160),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(160, 96, 5, padding=2), nn.Tanh(),nn.BatchNorm2d(96),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(96, 32, 5, padding=2), nn.Tanh(),nn.BatchNorm2d(32),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(32, 16, 5, padding=2), nn.Tanh(),nn.BatchNorm2d(16),
            nn.Conv2d(16, self.channel, 5, padding=2), nn.Tanh()
        )

        self.AEid = nn.Sequential(
            nn.Conv2d(self.channel, 16, 5, padding=2), nn.Tanh(),nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 5, padding=2), nn.Tanh(),nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 96, 5, padding=2), nn.Tanh(),nn.BatchNorm2d(96),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 160, 5, padding=2), nn.Tanh(),nn.BatchNorm2d(160),
            nn.MaxPool2d(2),
            nn.Conv2d(160, 256, 3, padding=1), nn.Tanh(),nn.BatchNorm2d(256),
            nn.Conv2d(256, 160, 3, padding=1), nn.Tanh(),nn.BatchNorm2d(160),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(160, 96, 5, padding=2), nn.Tanh(),nn.BatchNorm2d(96),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(96, 32, 5, padding=2), nn.Tanh(),nn.BatchNorm2d(32),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(32, 16, 5, padding=2), nn.Tanh(),nn.BatchNorm2d(16),
            nn.Conv2d(16, self.channel, 5, padding=2), nn.Tanh()
            )

        for block in self._modules:
            for m in block:
                weight_init_normal(m)

    def forward(self, x):
        if self.id == 1:
            x = self.share_layer(x)
            x1 = self.cls1(x)
            x2 = self.cls2(x1.view(x1.size(0), -1))
            x3 = self.AEmtl(x)
            return x2, x3
        elif self.id == 2:
            x = self.AEid(x)
            return x
        elif self.id == 3:
            x1 = self.share_layer(x)
            x_cls1 = self.cls1(x1)
            x_cls2 = self.cls2(x_cls1.view(x_cls1.size(0), -1))
            x_ae_mtl = self.AEmtl(x1)
            x_ae_id = self.AEid(x)
            x_distr = self.distr(x_cls1.view(x_cls1.size(0), -1) / self.T)
            return x_cls2, x_ae_id, x_ae_mtl, x_distr
        elif self.id == 4:
            x = self.share_layer(x)
            x1 = self.cls1(x)
            x2 = self.cls2(x1.view(x1.size(0), -1))
            return x2
        else:
            raise ValueError('Invalid Model Output')

    def set_id(self, id):
        self.id = id

    def set_T(self, T):
        self.T = T

    # def set_fix_param(self):
    #     if self.id == 3:
    #         for k, v in self.share_layer.named_parameters():
    #             v.requires_grad = False
    #
    # def reset_fix_param(self):
    #     if self.id == 3:
    #         for k, v in self.share_layer.named_parameters():
    #             v.requires_grad = True

    def reset_aeid_params(self):
        for m in self.AEid._modules:
            weight_init_normal(m)

class MnistMolel(nn.Module):
    def __init__(self, id=1, T=2.0, channel=1):
        super().__init__()
        if id not in [1, 2, 3, 4]:
            raise ValueError('Invalid Model type')
            sys.exit()
        self.id = id
        self.channel = channel
        self.T =T
        # id = 1:mtl method
        # id = 2:autoencoder id
        # id = 3:Output all
        # id = 4:cls only for attack
        self.share_layer = nn.Sequential(
            nn.Conv2d(self.channel, 16, 5, padding=2), nn.Tanh(), nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2), nn.Tanh(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.Tanh(), nn.BatchNorm2d(64),
        )
        self.cls1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, padding=2), nn.Tanh(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 10, 4)
        )
        self.cls2 = nn.LogSoftmax(dim=1)
        self.distr = nn.Softmax(dim=1)

        self.AEmtl = nn.Sequential(
            nn.Conv2d(64, 32, 5, padding=2), nn.Tanh(), nn.BatchNorm2d(32),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(32, 16, 5, padding=2), nn.Tanh(), nn.BatchNorm2d(16),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(16, self.channel, 5, padding=2), nn.Tanh()
        )

        self.AEid = nn.Sequential(
            nn.Conv2d(self.channel, 16, 5, padding=2), nn.Tanh(), nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2), nn.Tanh(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.Tanh(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 5, padding=2), nn.Tanh(), nn.BatchNorm2d(32),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(32, 16, 5, padding=2), nn.Tanh(), nn.BatchNorm2d(16),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(16, self.channel, 5, padding=2), nn.Tanh()
            )

        for block in self._modules:
            for m in block:
                weight_init_normal(m)

    def forward(self, x):
        if self.id == 1:
            x = self.share_layer(x)
            x1 = self.cls1(x)
            x2 = self.cls2(x1.view(x1.size(0), -1))
            x3 = self.AEmtl(x)
            return x2, x3
        elif self.id == 2:
            x = self.AEid(x)
            return x
        elif self.id == 3:
            x1 = self.share_layer(x)
            x_cls1 = self.cls1(x1)
            x_cls2 = self.cls2(x_cls1.view(x_cls1.size(0), -1))
            x_ae_mtl = self.AEmtl(x1)
            x_ae_id = self.AEid(x)
            x_distr = self.distr(x_cls1.view(x_cls1.size(0), -1) / self.T)
            return x_cls2, x_ae_id, x_ae_mtl, x_distr
        elif self.id == 4:
            x = self.share_layer(x)
            x1 = self.cls1(x)
            x2 = self.cls2(x1.view(x1.size(0), -1))
            return x2
        else:
            raise ValueError('Invalid Model Output')

    def set_id(self, id):
        self.id = id

    def set_T(self, T):
        self.T = T

    def reset_aeid_params(self):
        for m in self.AEid._modules:
            weight_init_normal(m)

class Cifar10Molel(nn.Module):
    def __init__(self, id=1, T=2.0, channel=3):
        super().__init__()
        if id not in [1, 2, 3, 4]:
            raise ValueError('Invalid Model type')
            sys.exit()
        self.id = id
        self.channel = channel
        self.T =T
        # id = 1:mtl method
        # id = 2:autoencoder id
        # id = 3:Output all
        # id = 4:cls only for attack
        self.share_layer = nn.Sequential(
            nn.Conv2d(self.channel, 16, 5, padding=2), nn.Tanh(), nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 5, padding=2), nn.Tanh(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 96, 5, padding=2), nn.Tanh(), nn.BatchNorm2d(96),
        )
        self.cls1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(96, 160, 5, padding=2), nn.Tanh(), nn.BatchNorm2d(160),
            nn.MaxPool2d(2),
            nn.Conv2d(160, 256, 3, padding=1), nn.Tanh(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 10, 4)
        )
        self.cls2 = nn.LogSoftmax(dim=1)
        self.distr = nn.Softmax(dim=1)

        self.AEmtl = nn.Sequential(
            nn.Conv2d(96, 32, 5, padding=2), nn.Tanh(),nn.BatchNorm2d(32),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(32, 16, 5, padding=2), nn.Tanh(),nn.BatchNorm2d(16),
            nn.Conv2d(16, self.channel, 5, padding=2), nn.Tanh()
        )

        self.AEid = nn.Sequential(
            nn.Conv2d(self.channel, 16, 5, padding=2), nn.Tanh(),nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 5, padding=2), nn.Tanh(),nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 96, 5, padding=2), nn.Tanh(),nn.BatchNorm2d(96),
            nn.Conv2d(96, 32, 5, padding=2), nn.Tanh(),nn.BatchNorm2d(32),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(32, 16, 5, padding=2), nn.Tanh(),nn.BatchNorm2d(16),
            nn.Conv2d(16, self.channel, 5, padding=2), nn.Tanh()
            )

        for block in self._modules:
            for m in block:
                weight_init_normal(m)

    def forward(self, x):
        if self.id == 1:
            x = self.share_layer(x)
            x1 = self.cls1(x)
            x2 = self.cls2(x1.view(x1.size(0), -1))
            x3 = self.AEmtl(x)
            return x2, x3
        elif self.id == 2:
            x = self.AEid(x)
            return x
        elif self.id == 3:
            x1 = self.share_layer(x)
            x_cls1 = self.cls1(x1)
            x_cls2 = self.cls2(x_cls1.view(x_cls1.size(0), -1))
            x_ae_mtl = self.AEmtl(x1)
            x_ae_id = self.AEid(x)
            x_distr = self.distr(x_cls1.view(x_cls1.size(0), -1) / self.T)
            return x_cls2, x_ae_id, x_ae_mtl, x_distr
        elif self.id == 4:
            x = self.share_layer(x)
            x1 = self.cls1(x)
            x2 = self.cls2(x1.view(x1.size(0), -1))
            return x2
        else:
            raise ValueError('Invalid Model Output')

    def set_id(self, id):
        self.id = id

    def set_T(self, T):
        self.T = T

    def reset_aeid_params(self):
        for m in self.AEid._modules:
            weight_init_normal(m)

class ReLUModel(nn.Module):
    def __init__(self, id=1, T=2.0, channel=1):
        super().__init__()
        if id not in [1, 2, 3, 4]:
            raise ValueError('Invalid Model type')
            sys.exit()
        self.id = id
        self.channel = channel
        self.T = T
        # id = 1:mtl method
        # id = 2:autoencoder id
        # id = 3:Output all
        # id = 4:cls only for attack
        self.share_layer = nn.Sequential(
            nn.Conv2d(self.channel, 32, 3, padding=1), nn.ReLU(True),
            #nn.BatchNorm2d(16),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(True),
            #nn.BatchNorm2d(32),
            nn.Conv2d(64, 96, 3, padding=1), nn.ReLU(True),
            # nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 192, 3, padding=1), nn.ReLU(True),
            # nn.BatchNorm2d(16),
        )
        self.cls1 = nn.Sequential(
            nn.Conv2d(192, 192, 3, padding=1), nn.ReLU(True),
            # nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(192, 192, 3, padding=1), nn.ReLU(True),
            # nn.BatchNorm2d(160),
            nn.MaxPool2d(2),
            nn.Conv2d(192, 192, 3, padding=1), nn.ReLU(True),
            #nn.BatchNorm2d(256),
            nn.Conv2d(192, 10, 4), nn.ReLU(True)
        )
        self.cls2 = nn.LogSoftmax(dim=1)
        self.distr = nn.Softmax(dim=1)

        self.AEmtl = nn.Sequential(
            nn.Conv2d(192, 96, 3, padding=1), nn.ReLU(True),
            #nn.BatchNorm2d(160),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(96, 64, 3, padding=1), nn.ReLU(True),
            #nn.BatchNorm2d(96),
            #nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(True),
            #nn.BatchNorm2d(32),
            nn.Conv2d(32, self.channel, 3, padding=1), nn.Sigmoid()
        )

        self.AEid = nn.Sequential(
            nn.Conv2d(self.channel, 32, 3, padding=1), nn.ReLU(True),
            # nn.BatchNorm2d(16),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(True),
            # nn.BatchNorm2d(32),
            nn.Conv2d(64, 96, 3, padding=1), nn.ReLU(True),
            # nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 192, 3, padding=1), nn.ReLU(True),
            # nn.BatchNorm2d(16),
            nn.Conv2d(192, 96, 3, padding=1), nn.ReLU(True),
            # nn.BatchNorm2d(160),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(96, 64, 3, padding=1), nn.ReLU(True),
            # nn.BatchNorm2d(96),
            # nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(True),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, self.channel, 3, padding=1), nn.Sigmoid()
            )

        for block in self._modules:
            for m in block:
                weight_init_normal(m)

    def forward(self, x):
        if self.id == 1:
            x = self.share_layer(x)
            x1 = self.cls1(x)
            x2 = self.cls2(x1.view(x1.size(0), -1))
            x3 = self.AEmtl(x)
            return x2, x3
        elif self.id == 2:
            x = self.AEid(x)
            return x
        elif self.id == 3:
            x1 = self.share_layer(x)
            x_cls1 = self.cls1(x1)
            x_cls2 = self.cls2(x_cls1.view(x_cls1.size(0), -1))
            x_ae_mtl = self.AEmtl(x1)
            x_ae_id = self.AEid(x)
            x_distr = self.distr(x_cls1.view(x_cls1.size(0), -1) / self.T)
            return x_cls2, x_ae_id, x_ae_mtl, x_distr
        elif self.id == 4:
            x = self.share_layer(x)
            x1 = self.cls1(x)
            x2 = self.cls2(x1.view(x1.size(0), -1))
            return x2
        else:
            raise ValueError('Invalid Model Output')

    def set_id(self, id):
        self.id = id

    def set_T(self, T):
        self.T = T

    def reset_aeid_params(self):
        for m in self.AEid._modules:
            weight_init_normal(m)