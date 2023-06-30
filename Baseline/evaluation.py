#!/usr/bin/env python

#SBATCH --job-name=PowerEva

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

if not os.path.exists('./evaluation/'):
    os.makedirs('./evaluation/')
    
    
args = parser.parse_args()
args = FormulateArgs(args)

valid_loader, datainfo = GetDataLoader(args, 'valid', path='../dataset/')
test_loader , datainfo = GetDataLoader(args, 'test',  path='../dataset/')
pprint.pprint(datainfo)

results = torch.zeros([10,51,4])
balance = np.round(np.linspace(0,1,51),2)

for seed in range(10):
    for idx, pb in enumerate(balance):

        args.SEED = seed
        args.powerbalance = pb

        modelname = f"pNN_data:{datainfo['dataname']}_seed:{args.SEED}_Penalty:{args.powerestimator}_Factor:{args.powerbalance}"

        model = torch.load(f'./models/{modelname}', map_location=args.DEVICE)
        model.SetParameter('args', args)

        SetSeed(args.SEED)

        evaluator = Evaluator(args).to(args.DEVICE)

        for x,y in valid_loader:
            X_valid, y_valid = x.to(args.DEVICE), y.to(args.DEVICE)
        for x,y in test_loader:
            X_test, y_test = x.to(args.DEVICE), y.to(args.DEVICE)

        acc_valid, power_valid = evaluator(model, X_valid, y_valid)
        acc_test, power_test   = evaluator(model, X_test,  y_test)

        results[seed,idx,0] = acc_valid
        results[seed,idx,1] = power_valid.cpu().item()
        results[seed,idx,2] = acc_test
        results[seed,idx,3] = power_test.cpu().item()
        
torch.save(results, f"./evaluation/result_data:{datainfo['dataname']}")
