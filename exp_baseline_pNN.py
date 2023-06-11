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

    setup = f"baseline_model_pNN_data_{datainfo['dataname']}_seed_{args.SEED:02d}_lnc_{args.lnc}_ilnc_{args.ilnc}.model"
    print(f'Training setup: {setup}.')

    msglogger = GetMessageLogger(args, setup)
    msglogger.info(f'Training network on device: {args.DEVICE}.')
    msglogger.info(f'Training setup: {setup}.')
    msglogger.info(datainfo)

    if os.path.isfile(f'{args.savepath}/{setup}'):
        print(f'{setup} exists, skip this training.')
        msglogger.info('Training was already finished.')
    else:
        topology = [datainfo['N_feature']] + args.hidden + [datainfo['N_class']]
        msglogger.info(f'Topology of the network: {topology}.')

        pnn = B.pNN(topology, args).to(args.DEVICE)
        
        msglogger.info(f'Number of parameters that could be learned: {len(dict(pnn.named_parameters()).keys())}.')
        msglogger.info(dict(pnn.named_parameters()).keys())
        msglogger.info(f'Number of parameters that are learned in this experiment: {len(pnn.GetParam())}.')
        
        lossfunction = B.pNNLoss(args).to(args.DEVICE)
        optimizer = torch.optim.Adam(pnn.GetParam(), lr=args.LR)

        if args.PROGRESSIVE:
            pnn, best = train_pnn_progressive(pnn, train_loader, valid_loader, lossfunction, optimizer, args, msglogger, UUID=setup)
        else:
            pnn, best = train_pnn(pnn, train_loader, valid_loader, lossfunction, optimizer, args, msglogger, UUID=setup)

        if best:
            if not os.path.exists(f'{args.savepath}/'):
                os.makedirs(f'{args.savepath}/')
            torch.save(pnn, f'{args.savepath}/{setup}')
            msglogger.info('Training if finished.')
        else:
            msglogger.warning('Time out, further training is necessary.')