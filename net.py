import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


def conv_black(bn=False, **kwargs):
    layer = []
    layer.append(nn.Conv2d(**kwargs))
    if bn:
        layer.append(nn.BatchNorm2d(kwargs['out_channels']))
    layer.append(nn.LeakyReLU(0.01, inplace=True))
    return layer


def T_conv(bn=False, act='LeakyReLU', **kwargs):
    layer = []
    layer.append(nn.ConvTranspose2d(**kwargs))
    if bn:
        layer.append(nn.BatchNorm2d(kwargs['out_channels']))
    if act == 'LeakyReLU':
        layer.append(nn.LeakyReLU(0.01, inplace=True))
    else:
        layer.append(nn.Tanh())
    return layer


def weight_init(model: nn.Module):
    for m in model.children():
        if type(m) in [nn.Conv2d, nn.Linear, nn.ConvTranspose2d]:
            torch.nn.init.kaiming_normal_(m.weight)
        elif type(m) in [nn.BatchNorm2d, nn.BatchNorm2d]:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)


class CAE(nn.Module):
    def __init__(self, k=8, dropout=0.1):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(
            *conv_black(in_channels=1, out_channels=50, kernel_size=3, stride=2, padding=0),
            *conv_black(in_channels=50, out_channels=50, kernel_size=3, stride=2, padding=2),
        )
        self.en_out = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(6050, k),
            nn.Tanh(),
        )
        self.de_in = nn.Sequential(
            nn.Linear(k, 6050),
            nn.LeakyReLU(0.01, inplace=True),
        )
        # weight_init(self.encoder)
        self.decoder = nn.Sequential(
            *T_conv(in_channels=50, out_channels=50, kernel_size=3, stride=2, padding=2),
            *T_conv(act='tanh', in_channels=50, out_channels=1, kernel_size=4, stride=2, padding=0)
        )

    def predict(self, x):
        return F.softmax(self.en_out(self.encoder(x)), dim=1)
        # return self.encoder(x)

    def cal_loss(self, real, fake, middle, coefficient=0.1):

        loss0 = F.mse_loss(real, fake)
        class_loss = F.mse_loss(torch.zeros_like(middle), middle)

        return loss0 + class_loss

    def forward(self, x, train=True):
        # encoder
        encoder_out = self.encoder(x)
        en_out = self.en_out(encoder_out)
        # decoder
        de_in = self.de_in(en_out).reshape_as(encoder_out)
        de_out = self.decoder(de_in)
        if train:
            return self.cal_loss(x, de_out, en_out)

class CAE_2(nn.Module):
    def __init__(self, k=8, dropout=0.1):
        super(CAE_2, self).__init__()
        self.e1 = nn.Sequential(
            *conv_black(in_channels=1, out_channels=50, kernel_size=3, stride=2, padding=0)
        )

        self.e2 = nn.Sequential(
            *conv_black(in_channels=50, out_channels=50, kernel_size=3, stride=2, padding=2)
        )

        self.en_out = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(6050, k),
            nn.Tanh(),
        )

        self.de_in = nn.Sequential(
            nn.Linear(k, 6050),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.d1 = nn.Sequential(
            *T_conv(in_channels=50, out_channels=50, kernel_size=3, stride=2, padding=2)
        )

        self.d2 = nn.Sequential(
            *T_conv(act='tanh', in_channels=50, out_channels=1, kernel_size=4, stride=2, padding=0)
        )

    def predict(self, x):
        return F.softmax(self.en_out(self.e2(self.e1(x))), dim=1)
        # return self.encoder(x)

    def forward(self, x, train=True,coefficient=0.1):
        # encoder
        e_in = x
        e1_out = self.e1(e_in)
        e2_out = self.e2(e1_out)
        z = self.en_out(e2_out)
        d_in = self.de_in(z).reshape_as(e2_out)

        d1_out = self.d1(d_in)
        d2_out = self.d2(d1_out)
        if train:
            loss1 = F.mse_loss(e_in,d2_out)
            loss2 = F.mse_loss(e1_out,d1_out) + F.mse_loss(e2_out,d_in)
            loss3 = F.mse_loss(torch.zeros_like(z),z)
            return loss1 + loss2 * coefficient + loss3
