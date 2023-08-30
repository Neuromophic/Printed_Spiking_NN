#!/usr/bin/env python
#SBATCH --job-name=SG
#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out
#SBATCH --mail-user=hzhao@teco.edu
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --partition=sdil
#SBATCH --gres=gpu:1

import os
import sys
sys.path.append(os.getcwd())
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pickle
import numpy as np
import training
import config
# import matplotlib.pyplot as plt
import MyTransformer

models = ['gpt-nano', 'gpt-micro', 'gpt-mini', 'gopher-44m', 'gpt2']

seed  = int(sys.argv[1])
model_idx = int(sys.argv[2])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for lr in range(-2,-5,-1):

    exp_setup = f'{models[model_idx]}_lr_{lr}_seed_{seed}'
    print(f'The experiment setup is {exp_setup}.')

    if os.path.exists(f'./NNs/predictor_{exp_setup}'):
        pass
    else:
        data = torch.load(f'./data/dataset.ds')

        X_train = data['X_train'].to(device)
        Y_train = data['Y_train'].to(device)
        X_valid = data['X_valid'].to(device)
        Y_valid = data['Y_valid'].to(device)
        X_test  = data['X_test'].to(device)
        Y_test  = data['Y_test'].to(device)

        train_data = TensorDataset(X_train, Y_train)
        valid_data = TensorDataset(X_valid, Y_valid)
        test_data  = TensorDataset(X_test, Y_test)

        train_loader = DataLoader(train_data, batch_size=256)
        valid_loader = DataLoader(valid_data, batch_size=len(valid_data))
        test_loader  = DataLoader(test_data, batch_size=len(test_data))

        config.SetSeed(seed)

        model_config = MyTransformer.GPT.get_default_config()
        model_config.model_type = models[model_idx]
        model_config.block_size = X_train.shape[1]
        model = MyTransformer.GPT(model_config)

        lossfunction = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=10**lr)

        model, train_loss, valid_loss = training.train_nn(model, train_loader, valid_loader, lossfunction, optimizer, UUID=exp_setup)
        torch.save(model, f'./NNs/predictor_{exp_setup}')
        
        plt.figure()
        plt.plot(train_loss, label='train')
        plt.plot(valid_loss, label='valid')
        plt.savefig(f'./NNs/train_curve_{exp_setup}.pdf', format='pdf', bbox_inches='tight')
        plt.close()
