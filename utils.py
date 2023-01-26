import torch
import os
import math
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils import data
import lpips
os.environ['TORCH_HOME']='/home/Ling.cf/HHD/conda/miniconda3/torch-model'

def visualization(feature_maps, mean=True):
    if mean == True:
        feature_maps = feature_maps.mean(dim=1).squeeze().detach().cpu()
        plt.imshow(feature_maps, interpolation='bicubic')
    else:
        nums = feature_maps.size(1)
        nrows = int(math.sqrt(nums))
        ncols = nums // nrows + 1
        plt.figure()
        for i in range(nums):
            img = feature_maps[:, i].squeeze().detach().cpu()
            plt.subplot(nrows, ncols, i+1)
            plt.imshow(img, interpolation='bicubic')
    plt.show()

class MyDataset(data.Dataset):
    def __init__(self, path, len_input, begin=0, interval=1):
        self.data = torch.load(path)[begin:]
        self.len_seq = self.data.size(0)
        self.interval = interval
        self.len_input = len_input
        self.nodes = []

    def __len__(self):
        len_index = 0
        for i in range(self.interval):
            len_index += ((self.len_seq + (self.interval - 1) - i) // (self.len_input * self.interval))
            self.nodes.append(len_index)
        return len_index

    def __getitem__(self, item):
        for round in range(len(self.nodes)):
            if item < self.nodes[round]:
                if round - 1 >= 0:
                    item = item - self.nodes[round-1]
                break
        input_seq = [self.data[i:i+1, :] for i in range(round + item * self.len_input * self.interval, round + (item + 1) * self.len_input *self.interval, self.interval)]
        return torch.cat(input_seq, dim=0)

class util_of_lpips():
    def __init__(self, net):
        self.loss_fn = lpips.LPIPS(net=net)


    def calc_lpips(self, predict, target):
        device = predict.device
        self.loss_fn.to(device)
        dist01 = self.loss_fn.forward(predict, target, normalize=True)
        return dist01


class Loss():
    def __init__(self):
        self.cross_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def d_loss(self, logits_real, logits_fake):  # 判别器的 loss
        loss = torch.abs(logits_real-1).mean() +torch.abs(logits_fake+1).mean()
        return loss, logits_fake.mean().item(), logits_real.mean().item()

    def g_loss(self, logits_fake,logits_real):  # 生成器的 loss
        loss = torch.abs(logits_fake-1).mean()
        return loss, logits_fake.mean().item(), logits_real.mean().item()

    def MSELoss(self, predict, target, lambda_t, lambda_l, reduction='free'):
        assert reduction in ['free', 'fixed']
        if reduction == 'free':
            # cur_loss = torch.sqrt(torch.sum((predict-target) ** 2)) * lambda_t * lambda_l / predict.size(0)
            cur_loss = torch.sum((predict - target) ** 2) * lambda_t * lambda_l / predict.size(0)
        elif reduction == 'fixed':
            cur_loss = torch.mean((predict - target) ** 2 * 1000) * lambda_t * lambda_l
        return cur_loss

    def KLDivLoss(self, predict, target, lambda_t, lambda_l):
        predict = F.log_softmax(predict, dim=1)
        target = F.softmax(target, dim=1)
        loss = self.kl_loss(predict, target)
        return  loss * lambda_t * lambda_l

    def mean_var(self, predict, target, lambda_t, lambda_l):
        b, c, h, w = predict.size()
        p_mean = torch.mean(predict)
        t_mean = torch.mean(target)
        p_var = torch.var(predict)
        t_var = torch.var(target)
        loss = (p_mean - t_mean)**2 + (p_var - t_var) ** 2
        # print(loss)
        return loss * (c * h * w) * lambda_t * lambda_l


class LReLuL(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, negative_slop=0.01, positive_slop=0.01):
        grad = torch.ones_like(input, dtype=input.dtype, device=input.device)
        add = torch.zeros_like(input, dtype=input.dtype, device=input.device)
        add[input > 1] = 1 - positive_slop
        grad[input < 0] = negative_slop
        grad[input > 1] = positive_slop
        ctx.save_for_backward(input, grad)
        input = input * grad + add
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, grad = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input * grad

class MyReLU(torch.autograd.Function):
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_.clamp(min=0.5, max=0.99)
        return output

    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_ < 0.5] = 0
        grad_input[input_ > 0.99] = 0
        return grad_input

class FilterReLU(nn.Module):
    def __init__(self):
        super(FilterReLU, self).__init__()
        self.relu = MyReLU.apply

    def forward(self, x):
        x = self.relu(x)
        return x

class DLReLU(nn.Module):
    def __init__(self):
        super(DLReLU, self).__init__()
        self.relu = LReLuL.apply

    def forward(self, x):
        x = self.relu(x)
        return x

if __name__ == '__main__':
    ftm1 = torch.rand(1, 64, 96, 128)
    ftm2 = torch.rand(1, 64, 96, 128)*10
    loss_fc = Loss()
    loss1 = loss_fc.MSELoss(ftm1, ftm2, 1, 1)
    loss2 = loss_fc.KLDivLoss(ftm1, ftm2, 1, 1)
    loss3 = loss_fc.mean_var(ftm1, ftm2, 1, 1)
    print(loss1, loss2, loss3)