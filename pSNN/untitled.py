#!/usr/bin/env python
#SBATCH --job-name=EvapSNN
#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out
#SBATCH --mail-user=hzhao@teco.edu
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --partition=sdil
#SBATCH --gres=gpu:1


import sys
import os
from pathlib import Path
import pickle
import torch
import pprint
sys.path.append(os.getcwd())
sys.path.append(str(Path(os.getcwd()).parent))
from utils import *
from configuration import *
import numpy as np

if not os.path.exists('./evaluation/'):
    os.makedirs('./evaluation/')
    
args = parser.parse_args([])
# args.task = 'temporized'
# args.metric = 'temporal_acc'
args.DEVICE = 'cpu'
args.SoftEva = True
args = FormulateArgs(args)

results = torch.zeros([13,10,5])


for ds in range(13):
    valid_loader, datainfo = GetDataLoader(args, 'valid', path='../dataset')
    test_loader , datainfo = GetDataLoader(args, 'test',  path='../dataset')

    for seed in range(10):
        args.SEED = seed
        args.DATASET = ds

        modelname = f"model_pSNN_data_{datainfo['dataname']}_seed_{args.SEED:02d}.model"
        
        ALL = 0
        POS = 0
        
        if os.path.isfile(f'./models/{modelname}'):                
            print(modelname)
            
            model = torch.load(f'./models/{modelname}', map_location=args.DEVICE)
            model.UpdateArgs(args)
            
            ALL = ALL + model.model[0].theta.numel() + model.model[1].theta.numel()
            POS = POS + (model.model[0].theta > 0.).sum() + (model.model[1].theta > 0.).sum()
            
            SetSeed(args.SEED)

            evaluator = Evaluator(args).to(args.DEVICE)

            # Validation phase
            total_val_samples = 0
            total_val_acc = 0.0
            total_val_power = 0.0
            with torch.no_grad():
                for x, y in valid_loader:
                    X_valid, y_valid = x.to(args.DEVICE), y.to(args.DEVICE)

                    acc_valid_batch, power_valid_batch = evaluator(model, X_valid, y_valid)
                    batch_size = X_valid.size(0)

                    total_val_samples += batch_size
                    total_val_acc += acc_valid_batch * batch_size
                    total_val_power += power_valid_batch * batch_size

            valid_acc = total_val_acc / total_val_samples
            valid_power = total_val_power / total_val_samples

            # Test phase
            total_test_samples = 0
            total_test_acc = 0.0
            total_test_power = 0.0
            with torch.no_grad():
                for x, y in test_loader:
                    X_test, y_test = x.to(args.DEVICE), y.to(args.DEVICE)

                    acc_test_batch, power_test_batch = evaluator(model, X_test, y_test)
                    batch_size = X_test.size(0)

                    total_test_samples += batch_size
                    total_test_acc += acc_test_batch * batch_size
                    total_test_power += power_test_batch * batch_size

            test_acc = total_test_acc / total_test_samples
            test_power = total_test_power / total_test_samples


            results[ds,seed,0] = valid_acc
            results[ds,seed,1] = test_acc
            results[ds,seed,2] = valid_power
            results[ds,seed,3] = test_power
            results[ds,seed,4] = POS / ALL
            
        else:
            results[ds,seed,:] = float('nan')
            
K = 3  # for example

valid_acc_data = results[:, :, 0]

# Obtain the top K validation accuracies and their indices for each row
top_k_values, top_k_seeds = torch.topk(valid_acc_data, K, dim=1)

# Initialize the tensor to hold the best K results for each row
best_data_k = torch.zeros(valid_acc_data.size(0), K, results.size(2))

# Populate best_data_k using the top K seeds
for i in range(valid_acc_data.size(0)):
    for j in range(K):
        best_data_k[i, j] = results[i, top_k_seeds[i, j]]

results_mean = torch.mean(best_data_k, dim=1)
results_std = torch.std(best_data_k, dim=1)

final = torch.cat([results_mean, results_std], dim=1)

np.savetxt('./result_new.txt', final.detach().numpy(), fmt='%.9f', delimiter='\t')