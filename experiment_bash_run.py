import os
import itertools

#lr = [0.0001]
lr = [0.001]
start_epochs = [1]# 0.025 0.05 0.1"]
#start_epochs = [1]# 0.025 0.05 0.1"]
rest_epochs = [3]# 0.025 0.05 0.1"]
reg = [0.00048828125]
#reg = [0, 2**-12, 2**-9, 2**-7, 2**-5, 2**-3, 2**-1]
zero_fix = [True]
local = True

inc = [2]
bs = [7]
#start_sigma = [2**x for x in range(-3, 3)]
#end_sigma = [2**x for x in range(-3, 3)]
#reg_function = 'exponential'
#reg_function = 'linear_on_grad'
reg_function = 'linear'
start_sigma = [1.0]
end_sigma = [1.0]
baselines = False 
fixed_sigma_first = [False]
fixed_sigma_all = False
load_pretrained = False
sigma_join = ['keep_the_same_divide_by_10'] #, 'keep_the_same_divide_by_2', 'keep_the_same_divide_by_10']
sigma_join = ['allow_retraining']
sigma_join = ['std_mu']
#sigma_join = ['baseline']

data = 'cifar_reduced'
#data = 'imagenet'
#models = ['resnet34', 'resnet50', 'resnet152', 'densenet161', 'vit_base_patch16_224']
#models =  ['resnet18', 'resnet34', 'resnet50', 'densenet161']
#models = ['deit_small_patch16_224', 'deit_tiny_patch16_224']
#models = ['resnet18', 'resnet34', 'resnet50']
#models = ['resnet18']
models = ['resnet18_cifar10']
#data = 'cifar10'
params = list(itertools.product(*[lr, start_epochs, rest_epochs, reg, zero_fix, start_sigma, end_sigma, inc, bs, fixed_sigma_first, models, sigma_join]))
for param_set in params:
         print('param_set', param_set)
         lr, se, re, r, zf, ss, es, inci, b, fsf, m, sj = param_set
         script_test =f'--model {m} --reg_function {reg_function} --data {data} --lr {lr} --start_epochs {se} --rest_epochs {re} --reg {r}  --start_sigma {ss}  --end_sigma {es} --inc {inci} --b {b} --sigma_join {sj} --want_to_save --prior' 
         if zf:
            script_test += ' --zero_fix'
         if local:
            script_test += ' --local'
         if load_pretrained:
            script_test += ' --load_pretrained'
         if baselines:
            script_test += ' --baselines'
         if fsf:
            script_test += ' --fixed_sigma_first'
         if fixed_sigma_all:
            script_test += ' --fixed_sigma_all'
         os.system('sbatch bayes_script.sh '+ script_test)
         print('sbatch bayes_script.sh '+ script_test)
