import torch
import math

def training(nn, loss_fn, optimizer, X_train, y_train, X_valid, y_valid, X_test, y_test):
    early_stop = False
    best_loss = math.inf
    patience = 0
    epoch = 0
    
    while not early_stop:
        y_pred_train = nn(X_train)
        loss_train = loss_fn(y_pred_train, y_train)
        acc_train = (y_pred_train.argmax(dim=1) == y_train).float().mean()
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred_valid = nn(X_valid)
            loss_valid = loss_fn(y_pred_valid, y_valid)
            acc_valid = (y_pred_valid.argmax(dim=1) == y_valid).float().mean()

            y_pred_test = nn(X_test)
            acc_test = (y_pred_test.argmax(dim=1) == y_test).float().mean()

        if loss_valid < best_loss:
            best_loss = loss_valid
            best_nn = nn
            patience = 0
        else:
            patience += 1
        
        if patience > 500:
            early_stop = True
            break
        
        if (epoch % 100)==0:
            print(f'epoch: {epoch:-8d} | train loss: {loss_train:.5e} | valid loss: {loss_valid:.5e} | train acc: {acc_train:.4f} | valid acc: {acc_valid:.4f} | test acc: {acc_test:.4f} | patience: {patience}')
        
        epoch += 1
        
    if early_stop:
        return best_nn
    else:
        return False
    
class lstm(torch.nn.Module):
    def __init__(self, N_channel, N_class):
        super().__init__()
        self.rnn = torch.nn.LSTM(N_channel, N_class, 1, batch_first=True)
        self.mac = torch.nn.Linear(N_class, N_class)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, X):
        X = X.permute(0,2,1)
        output, (h,c) = self.rnn(X)
        weighted_sum = self.mac(output[:,-1,:])
        result = self.softmax(weighted_sum)
        return result
    
    
class cnn(torch.nn.Module):
    def __init__(self, N_channel, N_length, N_class):
        super().__init__()
        
        stride = max(2,int(N_class/2))
        size = N_class
        padding = int((N_class-1)/2)

        L_in = N_length
        L_outs = []
        while L_in > N_class:
            L_out = int((L_in+2*padding-size) / stride) + 1
            L_outs.append(L_out)
            L_in = L_out
        L_outs.pop()
        
        N_channels = torch.linspace(N_channel, 1, len(L_outs)+1).round().long()

        self.model = torch.nn.Sequential()
        for i in range(len(L_outs)):
            self.model.add_module(f'{i}_conv', torch.nn.Conv1d(N_channels[i], N_channels[i+1], size, stride=stride, padding=padding))
            self.model.add_module(f'{i}_act', torch.nn.PReLU())
        self.model.add_module(f'{i+1}', torch.nn.Flatten())
        self.model.add_module(f'{i+2}', torch.nn.Linear(L_outs[-1],N_class))
        self.model.add_module(f'{i+3}', torch.nn.Softmax(dim=1))

    def forward(self, X):
        return self.model(X)