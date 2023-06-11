#!/usr/bin/env python
#SBATCH --job-name=BSL
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
sys.path.append(os.path.join(os.getcwd(), 'utils'))
from configuration import *
import torch
import pprint
from utils import *

args = parser.parse_args()

if args.ilnc:
    import BaselineModelsILNC as B
else:
    import BaselineModels as B

for seed in range(50):

    args.SEED = seed
    args = FormulateArgs(args)
    
    print(f'Training network on device: {args.DEVICE}.')
    MakeFolder(args)

    train_loader, datainfo = GetDataLoader(args, 'train')
    valid_loader, datainfo = GetDataLoader(args, 'valid')
    test_loader, datainfo = GetDataLoader(args, 'test')
    pprint.pprint(datainfo)

    SetSeed(args.SEED)

    setup = f"baseline_model_LSTM_data_{datainfo['dataname']}_seed_{args.SEED:02d}_lnc_{args.lnc}_ilnc_{args.ilnc}.model"
    print(f'Training setup: {setup}.')

    msglogger = GetMessageLogger(args, setup)
    msglogger.info(f'Training network on device: {args.DEVICE}.')
    msglogger.info(f'Training setup: {setup}.')
    msglogger.info(datainfo)

    if os.path.isfile(f'{args.savepath}/{setup}'):
        print(f'{setup} exists, skip this training.')
        msglogger.info('Training was already finished.')
    else:
        lstm = B.lstm(args, datainfo['N_feature'], datainfo['N_class']).to(args.DEVICE)
        
        msglogger.info(f'Number of parameters that could be learned: {len(dict(lstm.named_parameters()).keys())}.')
        msglogger.info(dict(lstm.named_parameters()).keys())
        msglogger.info(f'Number of parameters that are learned in this experiment: {len(lstm.GetParam())}.')
        
        lossfunction = B.CELOSS().to(args.DEVICE)
        optimizer = torch.optim.Adam(lstm.GetParam(), lr=args.LR)

        if args.PROGRESSIVE:
            lstm, best = train_pnn_progressive(lstm, train_loader, valid_loader, lossfunction, optimizer, args, msglogger, UUID=setup)
        else:
            lstm, best = train_pnn(lstm, train_loader, valid_loader, lossfunction, optimizer, args, msglogger, UUID=setup)

        if best:
            if not os.path.exists(f'{args.savepath}/'):
                os.makedirs(f'{args.savepath}/')
            torch.save(lstm, f'{args.savepath}/{setup}')
            msglogger.info('Training if finished.')
        else:
            msglogger.warning('Time out, further training is necessary.')