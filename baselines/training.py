import torch
import math

def training(nn, loss_fn, optimizer, X_train, y_train, X_valid, y_valid, X_test, y_test):
    early_stop = False
    best_loss = math.inf
    patience = 0

    for epoch in range(100000):
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
        
        print(f'epoch: {epoch:-8d} | train loss: {loss_train:.5e} | valid loss: {loss_valid:.5e} | train acc: {acc_train:.4f} | valid acc: {acc_valid:.4f} | test acc: {acc_test:.4f} | patience: {patience}')

    if early_stop:
        return best_nn
    else:
        return False