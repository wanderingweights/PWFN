import lightning as pl
import timm
import torch
from models.Bayes_Modules import Convertor, BayesConv2d, BayesLinear
import numpy as np
from torchmetrics import Accuracy
from cifar10_models.resnet import resnet18 as cifar10_resnet18
import torch.nn.functional as  F
from scipy.stats import entropy
from torch.optim.lr_scheduler import StepLR

from torchvision.models import resnet18

class Bayesian_WFN(pl.LightningModule):
    def __init__(self, sig_w, lr = 0.1, num_of_in_ch = 1, model = None, reg_function ='linear', num_classes = 1000, prior = True):
        super().__init__()
        self.lr = lr
        self.model = self.select_model(model)
        Convertor.orig_to_bayes(self.model, prior)
        self.sig_w = sig_w
        self.sigma_params = [n for (n, i) in self.named_parameters() if 'sigma' in n]
        self.vaccuracy = Accuracy(task = 'multiclass', num_classes = num_classes)
        self.ttaccuracy = Accuracy(task = 'multiclass', num_classes = num_classes)
        self.ttaccuracy5 = Accuracy(task = 'multiclass', num_classes = num_classes, top_k = 5)
        self.reg_function = self.select_reg_function(reg_function)
        self.save_hyperparameters() 
    
    def select_model(self, model_name):
        if model_name is None or model_name == 'resnet18_cifar10':
            model = cifar10_resnet18()
            model.load_state_dict(torch.load('cifar10_models/state_dicts/resnet18.pt'))
        else:
            model = timm.create_model(model_name, pretrained=True)
        return model

    def get_summary_sigma(self):
        try:
            sigmas = torch.cat([torch.flatten(p) for n, p in self.named_parameters() if 'sigma' in n])
            return torch.mean(sigmas), torch.max(sigmas), torch.min(sigmas), torch.std(sigmas)
        except:
             return 0, 0, 0, 0

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        cel = F.cross_entropy(y_hat, y)
        self.log("train_cel", cel)
        return cel

    def only_update_sigmas(self):
        for n, p in self.named_parameters():
            if 'sigma' in n:
                p.requires_grad = True
            elif 'mu' in n:
                p.requires_grad = False

    def allow_update_to_mus(self):
        for n, p in self.named_parameters():
            p.requires_grad = True
#            if 'sigma' in n:
#                p.requires_grad = True
#            else:
#                p.requires_grad = True


    def get_nearest_power_of_two(self, x):
        return torch.sign(x) * 2**(torch.clamp(torch.round(x), min=-7) - 1)

    def linear_reg(self, x, notfixed = None):
        if notfixed is None:
            notfixed = torch.ones_like(x.data, dtype = torch.bool)
            out = torch.nn.functional.relu((0.05 - x.data))
            x.grad.data[notfixed] -= self.sig_w * out[notfixed] * torch.abs(x.eps[notfixed])
        else:
            x.grad.data -= self.sig_w * torch.nn.functional.relu((0.05 - x.data)) * torch.abs(x.eps)
        return x.grad.data 

    def linear_reg_on_grad(self, x, notfixed = None):
        # if fixed is none, then fixed is all of x 
        if notfixed is None:
            notfixed = torch.ones_like(x.data, dtype = torch.bool)
            weight = 1 - torch.abs(x.eps[notfixed])
            less_than_05 = x.data[notfixed] < 0.05
            flat = torch.flatten(x.grad.data[notfixed][less_than_05])
            flat_weight = torch.flatten(weight[less_than_05])
            sm = torch.nn.functional.softmin(flat_weight * torch.abs(flat)/0.1)
            x.grad.data[x.grad.data < 0] = torch.abs(x.grad.data[x.grad.data < 0])
            x.grad.data[x <= 2**-30] = 0
            x.grad.data[notfixed][less_than_05] -= sm * self.sig_w
        else:
            weight = 1 - torch.abs(x.eps)
            less_than_05 = x.data < 0.05
            flat = torch.flatten(x.grad.data[less_than_05])
            flat_weight = torch.flatten(weight[less_than_05])
            sm = torch.nn.functional.softmin(flat_weight * torch.abs(flat)/0.1)
            x.grad.data[x.grad.data < 0] = torch.abs(x.grad.data[x.grad.data < 0])
            x.grad.data[x <= 2**-30] = 0
            x.grad.data[less_than_05] -= sm * self.sig_w
        return x.grad.data
 
    def exp_reg(self, x):
        return (torch.exp(torch.nn.functional.relu(0.05 - x)) - 1)

    def select_reg_function(self, reg_function):
        if reg_function == 'linear':
            return self.linear_reg
        if reg_function == 'exponential':
            return self.exp_reg
        if reg_function == 'linear_on_grad':
            return self.linear_reg_on_grad
    
#    def on_after_backward(self):
#        with torch.no_grad():
#            for n, p in self.named_parameters():
#                if 'downsample.0.weight_sigma' in n or 'downsample.0' in n:
#                    print(n, torch.mean(p), torch.std(p), p[0,0])

    def on_after_backward(self):
         with torch.no_grad():
             for n, p in self.named_parameters():
                     if p.requires_grad and hasattr(p, 'is_fixed') and p.grad is not None:
                         p.grad.data[p.is_fixed] = torch.zeros_like(p.grad.data[p.is_fixed])
                         if 'weight_sigma' in n:
                            p.grad.data = self.reg_function(p, p.not_fixed) 
                     elif 'weight_sigma' in n and p.requires_grad and p.grad is not None:
                                 try:
                                    p.grad.data = self.reg_function(p, None)
                                 except Exception as e:
                                     print(n, ' has no eps', e)

    def freeze_sigma(self):
        for n, p in self.named_modules():
            if isinstance(p, (BayesConv2d, BayesLinear)):
                p.freeze()

    def unfreeze_sigma(self):
        for n, p in self.named_modules():
            if isinstance(p, (BayesConv2d, BayesLinear)):
                p.unfreeze()
 
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        celoss = F.cross_entropy(y_hat, y) 
        self.log('test_cel', celoss, logger=True,on_step=True,  on_epoch=True, sync_dist=True)
        sig_mean, sig_max, sig_min, sig_std = self.get_summary_sigma()
        self.log("test_sigma_mean", sig_mean, on_epoch=True, logger = False, on_step=True, sync_dist=True)
        self.log("test_sigma_max", sig_max, on_epoch=True, logger = False, on_step=True, sync_dist=True)
        self.log("test_sigma_min", sig_min, on_epoch=True, logger = False, on_step=True, sync_dist=True)
        self.log("test_sigma_std", sig_std, on_epoch=True, logger = False, on_step=True, sync_dist=True)
        return {'test_loss': celoss, 'pred': y_hat, 'target': y}

    
    def on_test_start(self):
        self.freeze_sigma()

    def on_test_end(self):
        self.unfreeze_sigma()

    def test_epoch_end(self, o):
        for out in o:
            self.ttaccuracy(F.softmax(out['pred'], dim =1), out['target'])
            self.ttaccuracy5(F.softmax(out['pred'], dim =1), out['target'])
        self.log('test_acc_epoch', self.ttaccuracy.compute())
        self.log('test_acc5_epoch', self.ttaccuracy5.compute())

    def forward(self, x):
        if isinstance(x, list):
            xin = x[0]
            out = self.model(xin)
            return (out, x[1])
        x = self.model(x)
        return x

    def get_weight_entropy(self):
        weights = []
        for n, p in self.named_parameters():
            if 'mu' in n and 'prior' not in n:
                weights.extend(torch.flatten(p).detach())
        weights = np.array(weights)
        v, c = np.unique(weights, return_counts = True)
        try:
                c_z = np.delete(c, v.index(0.))
        except:
	        c_z = c
#	        print('no zero entry')
        c = np.array(c) / np.sum(c) # including zeros
#        c_z = np.array(c_z) / np.sum(c_z) # not including zeros
        return entropy(c, base=2) #, entropy(c_z, base=2)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        cel = F.cross_entropy(y_hat, y)
        self.log("val_cel", cel, on_epoch=True, logger = False, on_step=True, sync_dist=True)
        sig_mean, sig_max, sig_min, sig_std = self.get_summary_sigma()
        self.log("sigma_mean", sig_mean, on_epoch=True, logger = False, on_step=True, sync_dist=True)
        self.log("sigma_max", sig_max, on_epoch=True, logger = False, on_step=True, sync_dist=True)
        self.log("sigma_min", sig_min, on_epoch=True, logger = False, on_step=True, sync_dist=True)
        self.log("sigma_std", sig_std, on_epoch=True, logger = False, on_step=True, sync_dist=True)
        return {'cel': cel, 'pred': y_hat, 'target': y}

    def validation_epoch_end(self, o):
        for out in o:
            self.vaccuracy(F.softmax(out['pred'], dim =1), out['target'])
        self.log('validation_acc_epoch', self.vaccuracy.compute(), prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
#        optim = torch.optim.Adam(self.parameters(), lr = self.lr)
        optim  = torch.optim.SGD(self.parameters(), lr = self.lr, weight_decay = 1e-4, momentum = 0.9)
        sched = StepLR(optim, step_size=10, gamma=0.1)
        if sched is None:
            return optim
        return [optim], [sched]

    def fix_nan(self):
        for p in self.parameters():
                p.data = torch.nan_to_num(p.data)


