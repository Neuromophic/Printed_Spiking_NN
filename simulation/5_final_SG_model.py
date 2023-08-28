import sys
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pickle
import numpy as np
import training
import config
import matplotlib.pyplot as plt
import os


seed  = int(sys.argv[1])
order  = int(sys.argv[2])
lr = -4
num_layer = 10

exp_setup = f'{num_layer}_{order}_{lr}_{seed}'
print(f'The experiment setup is {exp_setup}.')

if os.path.exists(f'./NNs/predictor_{exp_setup}'):
    pass
else:
    data = torch.load(f'./data/dataset_order_{order}.ds')

    X_train = data['X_train']
    Y_train = data['Y_train']
    X_valid = data['X_valid']
    Y_valid = data['Y_valid']
    X_test  = data['X_test']
    Y_test  = data['Y_test']

    train_data = TensorDataset(X_train, Y_train)
    valid_data = TensorDataset(X_valid, Y_valid)
    test_data  = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(train_data, batch_size=256)
    valid_loader = DataLoader(valid_data, batch_size=len(valid_data))
    test_loader  = DataLoader(test_data, batch_size=len(test_data))

    topology = (np.round(np.logspace(np.log(X_train.shape[1]),
                                        np.log(Y_train.shape[1]),
                                        num=num_layer, base=np.e))).astype(int)

    config.SetSeed(seed)
    model = torch.nn.Sequential()
    for t in range(len(topology)-1):
        model.add_module(f'{t}-MAC', torch.nn.Linear(topology[t], topology[t+1]))
        model.add_module(f'{t}-ACT', torch.nn.PReLU())

    lossfunction = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=10**lr)

    model, train_loss, valid_loss = training.train_nn(model, train_loader, valid_loader, lossfunction, optimizer, UUID=exp_setup)
    torch.save(model, f'./NNs/predictor_{exp_setup}')
    
    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.savefig(f'./NNs/train_curve_{exp_setup}.pdf', format='pdf', bbox_inches='tight')
    plt.close()
