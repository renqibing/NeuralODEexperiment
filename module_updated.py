import torch
from torch import nn
import torch.functional as F
import math
from torchdiffeq import odeint_adjoint as odeint

# downsample + feature layers + fc
num_classes = 10


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def conv3x3(input_planes, output_planes, stride=1, padding=1):
    return nn.Conv2d(input_planes, output_planes, kernel_size=3, stride=stride, padding=padding, bias=False)


def conv1x1(input_planes, output_planes, stride=1):
    return nn.Conv2d(input_planes, output_planes, kernel_size=1, stride=stride, padding=0)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes, 1)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_planes = 64
        # self.conv1 = conv3x3(3, self.in_planes, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.conv1 = nn.Sequential(
            conv3x3(3,self.in_planes,stride= 1,padding=1),
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU()
        )
        # for i in self.bn1.parameters():
        #     i.requires_grad = False

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # self.bn2 = nn.BatchNorm2d(512)
        # self.relu = nn.ReLU()
        self.fc = nn.Linear(512, num_classes)
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        # out = self.avgpool(out)
        # out = self.relu(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # out = self.relu(out)
        out = self.fc(out)

        return out

    def _make_layer(self, block, planes, blocks, stride=1):
        out_planes = planes * block.expansion
        downsample = None
        if stride != 1 or out_planes != self.in_planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        # if downsample is not None:
            # for i in downsample._modules['1'].parameters():
                # i.requires_grad = False

        layers = []
        layers.append(block(self.in_planes, out_planes, stride, downsample))
        self.in_planes = out_planes
        for i in range(1, blocks):
            layer = block(self.in_planes, planes)
            layers.append(layer)
        return nn.Sequential(*layers)


class ConcatConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, stride=1, padding=1, bias=False):
        super().__init__()
        self._layer = conv3x3(dim_in + 1, dim_out, stride=stride, padding=padding)

    def forward(self, x, t):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        tx = torch.cat((x, tt), 1)
        return self._layer(tx)


class ODEfunc(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        dim = in_planes
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        # self.norm1 = nn.BatchNorm2d(dim)
        # self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(dim, dim)
        self.bn2 = nn.BatchNorm2d(dim)
        self.conv2 = conv3x3(dim, dim)
        self.bn3 = nn.BatchNorm2d(dim)
        self.conv3 = conv3x3(dim, dim)
        self.bn4 = nn.BatchNorm2d(dim)
        self.conv4 = conv3x3(dim, dim)
        # self.bn5 = nn.BatchNorm2d(dim)
        # self.conv1 = ConcatConv2d(64, 128, 1, 1)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.conv2 = ConcatConv2d(128, 256, 1, 1)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.conv3 = ConcatConv2d(256, 512, 1, 1)
        # self.bn4 = nn.BatchNorm2d(512)
        # self.conv4 = ConcatConv2d(512, 256, 1, 1)
        # self.conv5 = ConcatConv2d(256, 128, 1, 1)
        # self.conv6 = ConcatConv2d(128, 64, 1, 1)

        # self.conv1 = conv3x3(64, 128, 1, 1)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.conv2 = conv3x3(128, 256, 1, 1)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.conv3 = conv3x3(256, 512, 1, 1)
        # self.bn4 = nn.BatchNorm2d(512)
        # self.conv4 = conv3x3(512, 256, 1, 1)
        # self.conv5 = conv3x3(256, 128, 1, 1)
        # self.conv6 = conv3x3(128, 64, 1, 1)
        self.nfe = torch.Tensor(1).zero_().cuda()

    def forward(self, t, x):
        self.nfe += 1
        # out = self.bn1(x)
        # out = self.relu(out)
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)
        # out = self.conv5(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        # out = self.conv6(out)
        # out = self.bn1(out)
        # out = self.relu(out)
        return out


class ODEblock(nn.Module):
    def __init__(self, odefunc, rtol=1e-3, atol=1e-3):
        super().__init__()
        self.integration_time = torch.tensor([0, 1]).float()
        self.odefunc = odefunc
        self.atol = atol
        self.rtol = rtol
        self.output = None

    def forward(self, x, t=None):
        if t is not None:
            time = t
        else:
            time = self.integration_time
        self.output = odeint(self.odefunc,
                             x,
                             time,
                             rtol=self.rtol,
                             atol=self.atol)
        return self.output[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class ODEnet(nn.Module):
    def __init__(self, rtol=1e-6, atol=1e-6):
        super().__init__()
        self.t = torch.Tensor([0, 1]).float()
        self.conv1 = conv3x3(3, 64, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace= True)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc1 = nn.Linear(1024,512)
        self.odefunc = ODEfunc(64)
        self.odeblock = ODEblock(self.odefunc, rtol=rtol, atol=atol)
        self.avgpool_1 = nn.AvgPool2d(stride=4,kernel_size=4)
        self.avgpool = nn.AvgPool2d(stride=2,kernel_size=2)

    def forward(self, x, t=None):
        # if t is not None:
        #     time = t
        # else:
        #     time = self.t

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.odeblock(out, t)
        # out = self.relu(out)
        # print("the output of odeblock is {}".format(out.shape))
        out = self.avgpool(out)
        out = self.relu(out)
        # print("the output of  is {}".format(out.shape))
        out = self.avgpool(out)
        out = self.relu(out)
        # print("the output of odeblock is {}".format(out.shape))
        out = self.avgpool(out)
        out = self.relu(out)

        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out


class ODEnetRandTime(nn.Module):
    def __init__(self, min_end_time = 1,max_end_time = 10,rtol=1e-6, atol=1e-6):
        super().__init__()
        self.min_end_time = min_end_time
        self.max_end_time = max_end_time
        self.t = torch.Tensor([0,1]).float()
        self.conv1 = conv3x3(3, 64, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace= True)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc1 = nn.Linear(1024,512)
        self.odefunc = ODEfunc(64)
        self.odeblock = ODEblock(self.odefunc, rtol=rtol, atol=atol)
        self.avgpool_1 = nn.AvgPool2d(stride=4,kernel_size=4)
        self.avgpool = nn.AvgPool2d(stride=2,kernel_size=2)

    def forward(self, x, t=None):
        if t is None:
            end_time = torch.rand(1)*(self.max_end_time - self.min_end_time) + self.min_end_time
            self.t = torch.Tensor([0,end_time.item()]).float()
            print(self.t)
            time = self.t
        else:
            time = self.t

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.odeblock(out, time)
        # out = self.relu(out)
        # print("the output of odeblock is {}".format(out.shape))
        out = self.avgpool(out)
        out = self.relu(out)
        # print("the output of  is {}".format(out.shape))
        out = self.avgpool(out)
        out = self.relu(out)
        # print("the output of odeblock is {}".format(out.shape))
        out = self.avgpool(out)
        out = self.relu(out)

        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out



# data = torch.randn(32,3,32,32).cuda()
# net = ODEnet().cuda()
# output = net(data)
# print(output)

#
import imageio

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# # # %%
# f = ODEfunc(64).to(device).to(device)  # shape of output like 32 x 512 x 4 x 4
# model = ResNet(ResBlock, [2, 2, 2, 2]).to(device)
# img1 = torch.randn(32, 64, 32, 32).to(device)
# time = torch.Tensor([0]).float().to(device)
# out = f(time, img1)
# print(out.shape)
# # print(nn.utils.parameters_to_vector(f.parameters()).shape)
# print(count_parameters(model))
#
# print(count_parameters(f))
# # with torch.no_grad:
# test = torch.randn(32, 3, 32, 32).to(device)
# t = torch.Tensor([0, 1]).float().to(device)
# odenet = ODEnet().to(device)
# # output = odenet(test, t)
# # print(output.shape)
# # print(count_parameters(odenet))
#
# res_output = model(test)
# print(res_output.shape)

#
# with torch.no_grad():
#     inp = torch.randn(1,3,32,32).to(device)
#     # f(inp,0).shape
#     t = torch.linspace(0,500,100).to(device)
#     odenet = ODEblock(f, rtol=1e-3, atol=1e-3).to(device).eval()
#     odenet(inp, t).shape
#     print(odenet.nfe)
#     print(odenet.outputs.shape)
#
# with torch.no_grad():
#     ims = odenet.outputs
#     ims = torch.sigmoid(ims)
#     ims = ims.squeeze().cpu().detach().numpy().transpose([0,2,3,1])
#     ims.shape
#     imageio.mimwrite("test.gif", ims, duration=0.05)

# with torch.no_grad():
#     ims2 = []
#     inp = torch.randn(1,3,32,32)
#     for i in range(100):
#         inp += f(0,inp)
#         print(inp.std())
#         ims2.append(torch.sigmoid(inp).squeeze().detach().numpy().transpose([1,2,0]))
#
#     imageio.mimwrite("conving.gif", ims2)

# torch.eig(torch.randn(784,784))
#
# odenet = ODEnet(tol=1e-6)
#
# test_in = torch.randn(32,3,32,32)
# test_out = odenet(test_in)
# nn.utils.parameters_to_vector(odenet.parameters()).shape
# t = torch.linspace(0,1,100)
# test_out = odenet(test_in, t=t)
#
# # import pytorch_utils.sacred_trainer as st
# loader_test = ((torch.randn(32,3,32,32), torch.randint(0,10, (32,))) for _ in range(32))
# # odenet.load_state_dict(torch.load("ODEMnistClassification\\12\\epoch001_24-12_0026_.statedict.pkl"))
#
# # import training_functions as tf
# tf.validate(odenet.cpu(), loader_test)
# odenet.train()
