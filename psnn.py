import torch
import torch.nn as nn


class SpikingNeuron(nn.Module):

    instances = []

    def __init__(
        self,
        threshold=1.0,
        init_hidden=False
    ):
        super(SpikingNeuron, self).__init__()

        SpikingNeuron.instances.append(self)
        self.init_hidden = init_hidden
        self.threshold = nn.Parameter(threshold)

    def spike_grad(self, mem_shift):
        forward = (mem_shift > 0).float()
        backward = torch.sigmoid(mem_shift * 5.)
        return forward.detach() + backward - backward.detach()

    def fire(self, mem):
        mem_shift = mem - self.threshold
        spk = self.spike_grad(mem_shift)
        return spk

    def mem_reset(self, mem):
        mem_shift = mem - self.threshold
        reset = self.spike_grad(mem_shift).clone().detach()
        return reset



class LIF(SpikingNeuron):
    def __init__(
        self,
        beta,
        threshold=1.0,
        init_hidden=False
    ):
        super().__init__(
            threshold,
            init_hidden
        )

        self.beta = nn.Parameter(beta)

    @staticmethod
    def init_leaky():
        mem = _SpikeTensor(init_flag=False)

        return mem


class _SpikeTensor(torch.Tensor):
    """Inherits from torch.Tensor with additional attributes.
    ``init_flag`` is set at the time of initialization.
    When called in the forward function of any neuron, they are parsed and
    replaced with a torch.Tensor variable.
    """

    @staticmethod
    def __new__(cls, *args, init_flag=False, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        *args,
        init_flag=True,
    ):
        self.init_flag = init_flag


def _SpikeTorchConv(*args, input_):
    """Convert SpikeTensor to torch.Tensor of the same size as ``input_``."""

    states = []
    if (
        len(args) == 1 and type(args) is not tuple
    ):  # if only one hidden state, make it iterable
        args = (args,)
    for arg in args:
        arg = arg.to("cpu")
        arg = torch.Tensor(arg)  # wash away the SpikeTensor class
        arg = torch.zeros_like(input_, requires_grad=True)
        states.append(arg)
    if len(states) == 1:  # otherwise, list isn't unpacked
        return states[0]

    return states


class Leaky(LIF):

    def __init__(
        self,
        beta,
        threshold=1.0,
        init_hidden=False
    ):
        super(Leaky, self).__init__(
            beta,
            threshold,
            init_hidden
        )

        if self.init_hidden:
            self.mem = self.init_leaky()

    def forward(self, input_, mem=False):

        if hasattr(mem, "init_flag"):  # only triggered on first-pass
            mem = _SpikeTorchConv(mem, input_=input_)
        elif mem is False and hasattr(
            self.mem, "init_flag"
        ):  # init_hidden case
            self.mem = _SpikeTorchConv(self.mem, input_=input_)
            print('init_hidden', self.init_hidden)
            print('mem', self.mem)


        if not self.init_hidden:
            self.reset = self.mem_reset(mem)
            mem = self._build_state_function(input_, mem)
            spk = self.fire(mem)

            return spk, mem

        # intended for truncated-BPTT where instance variables are hidden
        # states
        if self.init_hidden:
            self._leaky_forward_cases(mem)
            self.reset = self.mem_reset(self.mem)
            self.mem = self._build_state_function_hidden(input_)


            self.spk = self.fire(self.mem)

            return self.spk, self.mem


    def _base_state_function(self, input_, mem):
        base_fn = self.beta.clamp(0, 1) * mem + input_
        return base_fn

    def _build_state_function(self, input_, mem):
        state_fn = self._base_state_function(
            input_, mem - self.reset * self.threshold
        )
        return state_fn

    def _base_state_function_hidden(self, input_):
        base_fn = self.beta.clamp(0, 1) * self.mem + input_
        return base_fn

    def _build_state_function_hidden(self, input_):
        state_fn = (
            self._base_state_function_hidden(input_)
            - self.reset * self.threshold
        )
        return state_fn
    
    def _leaky_forward_cases(self, mem):
        if mem is not False:
            raise TypeError(
                "When `init_hidden=True`, Leaky expects 1 input argument."
            )