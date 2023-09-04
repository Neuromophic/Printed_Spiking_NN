#!/usr/bin/env python
#SBATCH --job-name=pSNN
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
from utils import *
import PrintedSpikingNN as pSNN
import pprint
import torch
from configuration import *


args = parser.parse_args()
args = FormulateArgs(args)

print(f'Training network on device: {args.DEVICE}.')
MakeFolder(args)

train_loader, datainfo = GetDataLoader(args, 'train')
valid_loader, datainfo = GetDataLoader(args, 'valid')
test_loader, datainfo = GetDataLoader(args, 'test')
pprint.pprint(datainfo)

SetSeed(args.SEED)

setup = f"model_pSNN_data_{datainfo['dataname']}_seed_{args.SEED:02d}.model"
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

    psnn = pSNN.PrintedSpikingNeuralNetwork(topology, args).to(args.DEVICE)

    lossfunction = pSNN.LFLoss(args).to(args.DEVICE)
    optimizer = torch.optim.Adam(psnn.GetParam(), lr=args.LR)

    if args.PROGRESSIVE:
        psnn, best = train_pnn_progressive(psnn, train_loader, valid_loader, lossfunction, optimizer, args, msglogger, UUID=setup)
    else:
        psnn, best = train_pnn(psnn, train_loader, valid_loader, lossfunction, optimizer, args, msglogger, UUID=setup)

    if best:
        if not os.path.exists(f'{args.savepath}/'):
            os.makedirs(f'{args.savepath}/')
        torch.save(psnn, f'{args.savepath}/{setup}')
        msglogger.info('Training if finished.')
    else:
        msglogger.warning('Time out, further training is necessary.')
