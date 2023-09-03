import numpy as np
import torch
import warnings

#===============================================================================
#============================ Single Spiking Neuron ============================
#===============================================================================
    
class SpikingNeuron(torch.nn.Module):
    def __init__(self, args, beta=None, threshold=None, random_state=True):
        super().__init__()
        self.args = args
        self.beta = beta
        self.threshold = threshold

        # whether to initialize the initial state randomly for simulating unknown previous state
        # this is especially useful for signals split by sliding windows
        self.random_state = random_state
    
    @property
    def feasible_beta(self):
        return torch.sigmoid(self.beta)
    
    @property
    def feasible_threshold(self):
        return torch.nn.functional.softplus(self.threshold)
    
    @property
    def DEVICE(self):
        return self.args.DEVICE
    
    def SpikeGenerator(self, surplus):
        # straight through estimator
        # forward is 0/1 spike
        forward = (surplus >= 0).float()
        # backward is sigmoid relaxation
        backward = torch.sigmoid(surplus * 5.)
        return forward.detach() + backward - backward.detach()
    
    def StateUpdate(self, x):
        # update the state of neuron with new input
        return self.feasible_beta * self.memory + x
    
    @property
    def fire(self):
        # detect whether the neuron fires
        return self.SpikeGenerator(self.memory - self.threshold)
    
    def StatePostupdate(self):
        # update the state of neuron after firing
        return self.memory - self.fire.detach() * self.threshold

    def SingleStepForward(self, x):
        self.memory = self.StateUpdate(x)
        self.spike = self.fire
        self.memory = self.StatePostupdate()
        return self.spike, self.memory
    
    def forward(self, x):
        N_batch, T = x.shape
        # initialize the initial memory to match the batch size
        if self.random_state:
            self.memory = torch.rand(N_batch).to(self.DEVICE) * self.threshold.detach()
        else:
            self.memory = torch.zeros(N_batch).to(self.DEVICE)
        # forward
        spikes = []
        memories = [self.memory] # add initial state
        for t in range(T):
            spike, memory = self.SingleStepForward(x[:,t])
            spikes.append(spike)
            memories.append(memory)
        memories.pop() # remove the last one to keep the same length as spikes
        # output
        return torch.stack(spikes, dim=1), torch.stack(memories, dim=1)
    
    def UpdateArgs(self, args):
        self.args = args    

#===============================================================================
#========================= Spiking Neurons in a Layer ==========================
#===============================================================================

class SpikingLayer(torch.nn.Module):
    def __init__(self, args, N_neuron, beta=None, threshold=None, random_state=True, spike_only=True):
        super().__init__()
        self.args = args
        
        # create a list of neurons, number of neurons is the number of channels
        self.SNNList = torch.nn.ModuleList()
        for n in range(N_neuron):
            self.SNNList.append(SpikingNeuron(args, beta, threshold, random_state))
        # whether to output only spikes, False is geneally for visualization
        self.spike_only = spike_only
        
    @property
    def DEVICE(self):
        return self.args.DEVICE
    
    def forward(self, x):
        spikes = []
        memories = []
        # for each channel, run the corresponding spiking neuron
        for c in range(x.shape[1]):
            spike, memory = self.SNNList[c](x[:,c,:])
            spikes.append(spike)
            memories.append(memory)
        if self.spike_only:
            return torch.stack(spikes, dim=1)
        else:
            return torch.stack(spikes, dim=1), torch.stack(memories, dim=1)
    
    def ResetOutput(self, spike_only):
        # reset the output mode
        self.spike_only = spike_only
        
    def UpdateArgs(self, args):
        self.args = args
        for neuron in self.SNNList:
            if hasattr(neuron, 'UpdateArgs'):
                neuron.UpdateArgs(args)

#===============================================================================
#==================== Weighted-sum for Temporal Signal =========================
#===============================================================================

class TemporalWeightedSum(torch.nn.Module):
    def __init__(self, args, N_in, N_out):
        super().__init__()
        self.args = args
        # this paameter contrains both weights and bias
        self.weight = torch.nn.Parameter(torch.randn(N_in+1, N_out)/10.)
    
    @property
    def DEVICE(self):
        return self.args.DEVICE
    
    def forward(self, x):
        # extend the input with 1 for bias
        x_extend = torch.cat([x, torch.ones(x.shape[0], 1, x.shape[2]).to(self.DEVICE)], dim=1)
        # for each time step, compute the weighted sum
        T = x.shape[2]
        result = []
        for t in range(T):
            mac = torch.matmul(x_extend[:, :, t], self.weight)
            result.append(mac)
        return torch.stack(result, dim=2)
    
    def UpdateArgs(self, args):
        self.args = args
    
#===============================================================================
#======================== Spiking Neural Network ===============================
#===============================================================================

class SpikingNeuralNetwork(torch.nn.Module):
    def __init__(self, args, topology, beta=None, threshold=None, random_state=True):
        super().__init__()
        self.args = args
        
        # initialize beta
        if beta is None:
            beta = torch.tensor(0.95)
        beta = torch.log(beta / (1 - beta))
        self.beta = torch.nn.Parameter(beta, requires_grad=True)
        
        # initialize threshold
        if threshold is None:
            threshold = torch.tensor(1.)
        self.threshold = torch.nn.Parameter(threshold, requires_grad=True)
        
        # create snn with weighted-sum and spiking neurons
        self.model = torch.nn.Sequential()
        for i in range(len(topology)-1):
            self.model.add_module('MAC'+str(i), TemporalWeightedSum(args, topology[i], topology[i+1]))
            self.model.add_module('SNNLayer'+str(i), SpikingLayer(args, topology[i+1], self.beta, self.threshold, random_state))
        self.InitOutput()
    
    @property
    def DEVICE(self):
        return self.args.DEVICE
    
    def forward(self, x):
        return self.model(x)

    def Power(self):
        # this function is just used to match the evaluation
        # the power of snn in-silico is acutally not considered
        return torch.zeros(1).to(self.DEVICE)

    def InitOutput(self):
        for layer in self.model:
            if hasattr(layer, 'ResetOutput'):
                layer.ResetOutput(True)
        self.model[-1].ResetOutput(False)

    def ResetOutput(self, Layer, spike_only=False):
        self.InitOutput()
        for l, layer in enumerate(self.model):
            if l==Layer:
                try:
                    layer.ResetOutput(spike_only)
                    print(f"The memory of layer {str(Layer)} will be output.")
                except:
                    print(f"Layer {str(Layer)} is not a spiking layer, it has no attribute 'ResetOutput'.")
    
    def UpdateArgs(self, args):
        self.args = args
        self.beta.args = args
        self.threshold.args = args
        for layer in self.model:
            if hasattr(layer, 'UpdateArgs'):
                layer.UpdateArgs(args)
                
    def GetParam(self):
        weights = [p for name, p in self.named_parameters() if name.endswith('weight')]
        nonlinear = [p for name, p in self.named_parameters() if name.endswith('beta') or name.endswith('threshold')]
        if self.args.lnc:
            return weights + nonlinear
        else:
            return weights

#===============================================================================
#============================= Loss Functin ====================================
#===============================================================================

class SNNLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def forward(self, model, input, label):
        spk, mem = model(input)
        L = []       
        for step in range(mem.shape[2]):
            L.append(self.loss_fn(mem[:,:,step], label))
        return torch.stack(L).mean()
        
    
