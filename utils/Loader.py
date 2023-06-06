import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append(os.path.join(os.getcwd()))
sys.path.append('../')

class dataset(Dataset):
    def __init__(self, dataset, args, datapath, mode='train'):
        self.args = args
        
        if datapath is None:
            datapath = os.path.join(args.DataPath, dataset)
        else:
            datapath = os.path.join(datapath, dataset)
        with open(datapath, 'rb') as f:
            data = pickle.load(f)
        
        X_train         = data['X_train']
        y_train         = data['y_train']
        X_valid         = data['X_valid']
        y_valid         = data['y_valid']
        X_test          = data['X_test']
        y_test          = data['y_test']
        
        if mode == 'train':
            self.X_train    = torch.cat([X_train for _ in range(args.R_train)], dim=0).to(args.DEVICE)
            self.y_train    = torch.cat([y_train for _ in range(args.R_train)], dim=0).to(args.DEVICE)
        elif mode == 'valid':
            self.X_valid    = torch.cat([X_valid for _ in range(args.R_train)], dim=0).to(args.DEVICE)
            self.y_valid    = torch.cat([y_valid for _ in range(args.R_train)], dim=0).to(args.DEVICE)
        elif mode == 'test':
            self.X_test    = torch.cat([X_test for _ in range(args.R_test)], dim=0).to(args.DEVICE)
            self.y_test    = torch.cat([y_test for _ in range(args.R_test)], dim=0).to(args.DEVICE)
        
        self.data_name  = data['name']
        self.N_class    = data['n_class']
        self.N_feature  = data['n_feature']
        self.N_train    = X_train.shape[0]
        self.N_valid    = X_valid.shape[0]
        self.N_test     = X_test.shape[0]

        self.mode = mode
        

    @property
    def noisy_X_train(self):
        noise = torch.randn(self.X_train.shape) * self.args.InputNoise + 1.
        return self.X_train * noise.to(self.args.DEVICE)

    @property
    def noisy_X_valid(self):
        noise = torch.randn(self.X_valid.shape) * self.args.InputNoise +1.
        return self.X_valid * noise.to(self.args.DEVICE)
    
    @property
    def noisy_X_test(self):
        noise = torch.randn(self.X_test.shape) * self.args.IN_test + 1.
        return self.X_test * noise.to(self.args.DEVICE)
    
    
    def __getitem__(self, index):
        if self.mode == 'train':
            x = self.noisy_X_train[index,:]
            y = self.y_train[index]
        elif self.mode == 'valid':
            x = self.noisy_X_valid[index,:]
            y = self.y_valid[index]
        elif self.mode == 'test':
            x = self.noisy_X_test[index,:]
            y = self.y_test[index]
        return x, y
    
    def __len__(self):
        if self.mode == 'train':
            return self.N_train * self.args.R_train
        elif self.mode == 'valid':
            return self.N_valid * self.args.R_train
        elif self.mode == 'test':
            return self.N_test * self.args.R_test
        
        
def GetDataLoader(args, mode, path=None):
    normal_datasets = ['Dataset_acuteinflammation.p',
                       'Dataset_balancescale.p',
                       'Dataset_breastcancerwisc.p',
                       'Dataset_cardiotocography3clases.p',
                       'Dataset_energyy1.p',
                       'Dataset_energyy2.p',
                       'Dataset_iris.p',
                       'Dataset_mammographic.p',
                       'Dataset_pendigits.p',
                       'Dataset_seeds.p',
                       'Dataset_tictactoe.p',
                       'Dataset_vertebralcolumn2clases.p',
                       'Dataset_vertebralcolumn3clases.p']

    split_manufacture = ['Dataset_acuteinflammation.p',
                         'Dataset_acutenephritis.p',
                         'Dataset_balancescale.p',
                         'Dataset_blood.p',
                         'Dataset_breastcancer.p',
                         'Dataset_breastcancerwisc.p',
                         'Dataset_breasttissue.p',
                         'Dataset_ecoli.p',
                         'Dataset_energyy1.p',
                         'Dataset_energyy2.p',
                         'Dataset_fertility.p',
                         'Dataset_glass.p',
                         'Dataset_habermansurvival.p',
                         'Dataset_hayesroth.p',
                         'Dataset_ilpdindianliver.p',
                         'Dataset_iris.p',
                         'Dataset_mammographic.p',
                         'Dataset_monks1.p',
                         'Dataset_monks2.p',
                         'Dataset_monks3.p',
                         'Dataset_pima.p',
                         'Dataset_pittsburgbridgesMATERIAL.p',
                         'Dataset_pittsburgbridgesSPAN.p',
                         'Dataset_pittsburgbridgesTORD.p',
                         'Dataset_pittsburgbridgesTYPE.p',
                         'Dataset_seeds.p',
                         'Dataset_teaching.p',
                         'Dataset_tictactoe.p',
                         'Dataset_vertebralcolumn2clases.p',
                         'Dataset_vertebralcolumn3clases.p']
    
    normal_datasets.sort()
    split_manufacture.sort()
    
    if path is None:
        path = args.DataPath
    
    datasets = os.listdir(path)
    datasets = [f for f in datasets if (f.startswith('Dataset') and f.endswith('.p'))]
    datasets.sort()

    
    if args.task=='normal':
        dataname = normal_datasets[args.DATASET]
        # data
        trainset  = dataset(dataname, args, path, mode='train')
        validset  = dataset(dataname, args, path, mode='valid')
        testset   = dataset(dataname, args, path, mode='test')

        # batch
        train_loader = DataLoader(trainset, batch_size=len(trainset))
        valid_loader = DataLoader(validset, batch_size=len(validset))
        test_loader  = DataLoader(testset,  batch_size=len(testset))
        
        # message
        info = {}
        info['dataname'] = trainset.data_name
        info['N_feature'] = trainset.N_feature
        info['N_class']   = trainset.N_class
        info['N_train']   = len(trainset)
        info['N_valid']   = len(validset)
        info['N_test']    = len(testset)
        
        if mode == 'train':
            return train_loader, info
        elif mode == 'valid':
            return valid_loader, info
        elif mode == 'test':
            return test_loader, info
    
    elif args.task=='split':
        train_loaders = []
        valid_loaders = []
        test_loaders  = []
        infos = []
        for dataname in split_manufacture:
            # data
            trainset  = dataset(dataname, args, mode='train')
            validset  = dataset(dataname, args, mode='valid')
            testset   = dataset(dataname, args, mode='test')
            # batch
            train_loaders.append(DataLoader(trainset, batch_size=len(trainset)))
            valid_loaders.append(DataLoader(validset, batch_size=len(validset)))
            test_loaders.append(DataLoader(testset,  batch_size=len(testset)))
            # message
            info = {}
            info['dataname'] = trainset.data_name
            info['N_feature'] = trainset.N_feature
            info['N_class']   = trainset.N_class
            info['N_train']   = len(trainset)
            info['N_valid']   = len(validset)
            info['N_test']    = len(testset)
            infos.append(info)

        return train_loaders, valid_loaders, test_loaders, infos
