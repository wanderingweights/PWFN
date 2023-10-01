import torch 
import json 
import numpy as np
import os
from Datasets.imagenet import ImageNet_Module
from models.Probablistic_WFN import Bayesian_WFN
import lightning as pl
from scipy.stats import entropy

# get all the models in the model_saves directory
# models = os.walk('model_saves/')
# print('here', models)
# # get the model names
# model_names = []
# for model in list(models):
#     # if any of the models are .pt files
#     for m in model[-1]:
#         if m.endswith('.pt') and 'baseline' not in m:
#             model_names.append(os.path.join(model[0], m))
#             print('model', model)

models = json.load(open('models.json', 'r'))

def get_the_entopy_and_unique_param_count(model):
    params = []
    # get the entropy of the model
    for n, p in model.named_parameters():
        if 'mu' in n and 'bn' not in n and 'prior' not in n:
            print(n)
            params.extend(p.flatten().tolist())
    # get the number of unique parameters
    unique_params, counts = np.unique(params, return_counts=True)
    probs = counts / len(params)
    return entropy(probs), len(unique_params)


data = ImageNet_Module(local = False)
data.setup()
number_of_tests = 20
for name in models:
    # check if substring is in name
    if not 'std_mu_sigma_resnet50' in name:
        continue
    if 'densenet' in name:
        model = Bayesian_WFN(2**-12, 0.1, 3, 'densenet161', 'linear')
    elif 'resnet50' in name:
        model = Bayesian_WFN(2**-12, 0.1, 3, 'resnet50', 'linear')
    elif 'resnet34' in name:
        model = Bayesian_WFN(2**-12, 0.1, 3, 'resnet34', 'linear')
    elif 'resnet18' in name: 
        model = Bayesian_WFN(2**-12, 0.1, 3, 'resnet18', 'linear')
    elif 'vit' in name:
        continue
        model = Bayesian_WFN(2**-12, 0.1, 3, 'vit_base_patch16_224', 'linear')
    elif 'deit_small' in name:
        continue
        model = Bayesian_WFN(2**-12, 0.1, 3, 'deit_small_patch16_224', 'linear')
    elif 'deit_tiny' in name:
        continue
        model = Bayesian_WFN(2**-12, 0.1, 3, 'deit_tiny_patch16_224', 'linear')

    results = {} 
    results['bins'] = {}
    before = {}
    try:
        model.load_state_dict(torch.load(models[name]), strict=False)
        results['entropy'], results['unique_params'] = get_the_entopy_and_unique_param_count(model)
        print(results['entropy'], results['unique_params'])
    except:
        print('failed to load ', name)
    trainer = pl.Trainer(accelerator='gpu', strategy='dp', devices=-1, max_epochs=0, enable_checkpointing=False, callbacks=None, gradient_clip_val=1)
    trainer.fit(model, data)
    res = trainer.test(model, dataloaders=data.test_dataloader())[0]
    results['standard_test'] = res
    model.to('cuda')
    # make predictions on the test set multiple times with different weights 
    predictions =  torch.zeros(len(data.predict_dataloader().dataset), number_of_tests, 1000)
    losses = torch.zeros(len(data.predict_dataloader().dataset), number_of_tests)
    true = torch.zeros(len(data.predict_dataloader().dataset), dtype=torch.long)

    loss = torch.nn.CrossEntropyLoss(reduction='none')
    batch_size = 128
    for i in range(number_of_tests):
        sample_pred = trainer.predict(model, data.predict_dataloader())
        for j, pred in enumerate(sample_pred):
            index = j * batch_size
            predictions[index:index+batch_size, i] = pred[0]
            losses[index:index+batch_size, i] = -loss(pred[0], pred[1])
            if i == 0:
                true[index: index+batch_size] = pred[1]
    # saves the predictions, losses and true values for each sample as an npz file back to the same location as the model but removes the .pt and adds _predictions.npz
    npz_save_path = models[name].replace('.pt', '_predictions.npz')
    if not os.path.exists(npz_save_path):
        # create the directory if it doesn't exist
        os.makedirs(os.path.dirname(npz_save_path), exist_ok=True)
    # save the predictions, losses and true values for each sample as an npz file
    np.savez(npz_save_path, predictions=predictions, losses=losses, true=true)

    # get the loss for each sample and take the softmax which will be the weighting for the predictions 
    loss_weight = torch.softmax(losses, dim=1)
    weighted_predictions = torch.argmax(torch.sum(predictions * loss_weight.unsqueeze(2), dim=1), dim=1)
    unweighted_predictions = torch.argmax(torch.sum(predictions, dim=1), dim=1)
    accuracy_mean_weighted = torch.sum(weighted_predictions == true) / len(true)
    accuracy_mean = torch.sum(unweighted_predictions == true) / len(true)
    results['accuracy_mean_loss_weighted'] = accuracy_mean.item()
    results['accuracy_mean'] = accuracy_mean_weighted.item()
    # what is the maximum prediction for each sample
    max_confidence = torch.max(torch.softmax(predictions, dim=2), dim=2).values
    # get the max confidence for each sample
    max_confidence = torch.max(max_confidence, dim=1)
    sorted_confidence = torch.argsort(max_confidence.values, dim=0)
    # bin the samples by their max confidence, 20 bins
    bin_index = torch.zeros(len(data.predict_dataloader().dataset))
    # assign each sample to a bin
    bin_i = 0
    for i in range(len(data.predict_dataloader().dataset)):
        if i % (len(data.predict_dataloader().dataset) // 20) == 0:
            bin_i += 1
        bin_index[sorted_confidence[i]] = bin_i - 1
    # what is the mean max confidence for each bin 
    bin_means = torch.zeros(20)
    for i in range(0, 20):
        # get the indices of the samples that are in bin i
        indices = torch.where(bin_index == i)
        # get the predictions for the samples in bin i
        bin_predictions = predictions[indices]
        bin_predictions = torch.argmax(bin_predictions, dim=2)
        bin_means[i] = torch.mean(max_confidence.values[indices])
        # get the true values for the samples in bin i
        bin_true = true[indices]
        # calculate the accuracy for the samples in bin i across all samples (dimension 2)
        bin_accuracy = 0
        for test in range(number_of_tests):
            test_bin_accuracy = torch.sum(bin_predictions[:, test] == bin_true) 
            bin_accuracy += test_bin_accuracy
        bin_accuracy = bin_accuracy / (number_of_tests * len(bin_true))
        results['bins'][i] = {'bin_accuracy': bin_accuracy.item(), 'bin_confidence': bin_means[i].item()}
        print(results['bins'][i], 'this is the result', i)
        print(results['accuracy_mean'], 'this is the mean accuracy')
    # save the results to a json file
    with open(name + '_results.json', 'w') as f:
        json.dump(results, f)

        


 
 