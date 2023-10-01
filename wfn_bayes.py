import random
import argparse
import os
import json
import torch 
import lightning as pl
from models.Probablistic_WFN import Bayesian_WFN
from clustering.Cluster import Cluster
import torch
import pandas as pd

from Datasets.cifar10 import CIFAR10DataModule
from Datasets.cifar10_reduced import CIFAR10ReducedDataModule
from Datasets.imagenet import ImageNet_Module


cluster_rounds = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 1.0]

def run_a_single_example(data, model, epochs, reg, dataref, reg_function, model_name, fixed_sigma, lr):
    if fixed_sigma:
        model.only_update_sigmas()
    trainer = pl.Trainer(accelerator='gpu', strategy='dp', devices=-1, max_epochs=epochs, enable_checkpointing=False, callbacks=None, gradient_clip_val=1)
    trainer.fit(model, data)
    path = f'model_saves/{reg_function}/{model_name}_{reg}_{dataref}_{lr}/'
    if not os.path.exists(path):
        os.mkdir(path)
    if fixed_sigma:
        model.allow_update_to_mus()
    torch.save(model.state_dict(), path + model_name + '.pt')
    res = trainer.test(model, dataloaders=data.test_dataloader())[0]
    val_res = trainer.test(model, dataloaders=data.val_dataloader())[0]
    save_baseline_results(reg, res, val_res, epochs, reg_function, model_name, lr)
    print('The results ', res, ' the reg ', reg)

def save_baseline_results(reg, t_res, v_res, epochs,  reg_function, model_name, lr):
    results = {}
    results['lr'] = lr
    results['reg'] = reg
    results['test_acc_epoch'] = t_res['test_acc_epoch']
#    results['test_loss_epoch'] = t_res['test_loss_epoch']
    results['val_loss_epoch'] = v_res['val_cel_epoch']
    results['epochs'] = epochs
    path = f'model_saves/{reg_function}/sgd_res18_{reg}_{model_name}_results.json'
    with open(path, 'w') as f:
        json.dump(results, f)


def save_model(model, reg, start_sig, end_sig, dataref, reg_function, hash):
    path = f'model_saves/{hash}_{dataref}/'
    # if the path does not exist, create it
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(model.state_dict(), path + 'model.pt')

def select_data(dataref, location):
    if dataref == 'cifar10':
        return CIFAR10DataModule()
    elif dataref == 'imagenet':
        return ImageNet_Module(data_dir= location)
    elif dataref == 'cifar_reduced':
         return CIFAR10ReducedDataModule()
    else:
        return None

def get_model_name(fixed_sigma):
    if fixed_sigma:
        return 'baseline_fixed'
    else:
        return 'baseline'

def learn_mus(model, data, epochs):
    model.freeze_sigma()
    model.allow_update_to_mus()
    trainer = pl.Trainer(accelerator='gpu', strategy='dp', devices=-1, max_epochs=epochs, enable_checkpointing=False, callbacks=None, gradient_clip_val=1)
    trainer.fit(model, data)
    return model, trainer

def learn_sigmas(model, data, epochs):
    model.unfreeze_sigma()
    model.only_update_sigmas()
    trainer = pl.Trainer(accelerator='gpu', strategy='dp', devices=-1, max_epochs=epochs, enable_checkpointing=False, callbacks=None, gradient_clip_val=1)
    trainer.fit(model, data)
    return model, trainer

def tiny_update_round(model, data, epochs):
    model.allow_update_to_mus()
    model.unfreeze_sigma()
    trainer = pl.Trainer(accelerator='gpu', strategy='dp', devices=-1, limit_train_batches= 0.05, max_epochs=epochs, enable_checkpointing=False, callbacks=None, gradient_clip_val=1)
    trainer.fit(model, data)
    return model, trainer

def learn_both_mu_sigma(model, data, epochs):
    model.allow_update_to_mus()
    model.unfreeze_sigma()
    trainer = pl.Trainer(accelerator='gpu', strategy='dp', devices=-1, max_epochs=epochs, enable_checkpointing=False, callbacks=None, gradient_clip_val=1)
    trainer.fit(model, data)
    return model, trainer

RUN_PATH = '/home/cc2u18/Bayes_WFN_Saves/'
def run_experiment(lr, zero_fix, first_epochs, epochs, reg, start_sigma, end_sigma, inc, b, want_to_save, dataref, reg_function, data_loc, fixed_sigma_first, fixed_sigma_all, load, model_type, prior, sigma_join):
            load = False
            print(lr, zero_fix, start_sigma, end_sigma, first_epochs, epochs, reg, want_to_save)
            data = select_data(dataref, data_loc)
            model = Bayesian_WFN(reg, num_of_in_ch= data.dims[0], lr = lr, model = model_type, reg_function=reg_function, num_classes = data.targets, prior=prior)
            model_name = get_model_name(fixed_sigma_first)
            cluster = Cluster(model, start_sigma=start_sigma, end_sigma=end_sigma, num_rounds=len(cluster_rounds), zero_fix = zero_fix, inc=inc, b = b, sigma_join=sigma_join)
            hashv = str(random.getrandbits(128))
            path = RUN_PATH + hashv + '/'
            model_path = f'model_saves/{reg_function}/sgd_res18_{reg}_{dataref}_{lr}/'
            if dataref == 'imagenet':
                    model.load_state_dict(torch.load(f'{model_path}/baseline.pt'))
            else:
                    model.load_state_dict(torch.load(f'model_saves/{reg_function}/{reg}/{model_name}.pt'))
            for i, r in enumerate(cluster_rounds):
                results = pd.DataFrame()
                if i == 0:
                    model, trainer = tiny_update_round(model, data, first_epochs)
                elif fixed_sigma_all: 
                    model, trainer = learn_mus(model, data, epochs)
                    model, trainer = learn_sigmas(model, data, epochs)
                else:
                    model, trainer  = learn_both_mu_sigma(model, data, epochs)
                try:
                    res = trainer.test(model, dataloaders=data.test_dataloader())[0]
                    clusters, percent, not_assigned, round_summary = cluster.cluster_round(r, i)
                    if os.path.exists(path + 'round_results.csv'):
                            round_summary.to_csv(path + 'round_results.csv', mode='a', header=False)
                    else:
                            os.mkdir(path)
                            round_summary.to_csv(path + 'round_results.csv')
                    entropy_val = model.get_weight_entropy()
                    r = {'model' : model_type, 'fixed_sigma_first': fixed_sigma_first, 'fixed_sigma_all': fixed_sigma_all, 'hash': str(hashv), 'reg_function': reg_function, 'b': b,  'inc': inc,  'epochs': epochs, 'first_epochs': first_epochs, 'start_sigma' :  start_sigma, 'end_sigma': end_sigma, 'zf': zero_fix, 'lr': lr, 'iteration': str(i), 'reg_w':reg, 'entropy': entropy_val, 'acc': res['test_acc_epoch'], 'round': r, 'percent': percent, 'num_clusters': len(clusters), 'prior': prior, 'sigma_join': sigma_join}
                    results = pd.concat([results, pd.DataFrame([r])])
                    if os.path.exists(f'csv_results/results_{data.name}.csv'):
                                results.to_csv(f'csv_results/results_{data.name}.csv', mode='a', header=False, index=False)
                    else:
                                results.to_csv(f'csv_results/results_{data.name}.csv', index=False)
                except:
                    print('failed to test')
            if want_to_save:
                    path = f'model_saves/{hashv}_{dataref}/'
                    if not os.path.exists(path):
                        os.mkdir(path)
                    torch.save(model.state_dict(), path + str(hashv)+ '.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default = 0.0001)
    parser.add_argument('--start_epochs', type=int, default=5)
    parser.add_argument('--rest_epochs', type=int, default=5)
    parser.add_argument('--reg', type=float, default=1.0)
    parser.add_argument('--b', type=float, default=7)
    parser.add_argument('--data', default='cifar10')
    parser.add_argument('--zero_fix', action='store_true')
    parser.add_argument('--start_sigma', type=float, default=1)
    parser.add_argument('--inc', type=float, default=1)
    parser.add_argument('--end_sigma', type=float, default=0.5)
    parser.add_argument('--reg_function', default='linear')
    parser.add_argument('--data_loc', default='/ssdfs/datasets/imagenet_2012/')
    parser.add_argument('--want_to_save', action='store_true')
    parser.add_argument('--fixed_sigma_first', action='store_true')
    parser.add_argument('--fixed_sigma_all', action='store_true')
    parser.add_argument('--load_pretrained', action='store_true')
    parser.add_argument('--prior', action='store_true')
    parser.add_argument('--model', default='deit_tiny_patch16_224')
    parser.add_argument('--sigma_join', default='mean_mu')
    args = parser.parse_args()
    run_experiment(args.lr, args.zero_fix, args.start_epochs, args.rest_epochs, args.reg, args.start_sigma, args.end_sigma, args.inc, args.b, args.want_to_save, args.data, args.reg_function, args.data_loc, args.fixed_sigma_first, args.fixed_sigma_all, args.load_pretrained, args.model, args.prior, args.sigma_join)

