import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score


def eval_acc(model, loader, device):
    model.eval()
    num_correct1 = 0
    num_correct2 = 0
    num_data1 = 0
    num_data2 = 0
    for s, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        target1 = targets[:,0]
        target2 = targets[:,1]
        [out1,out2,frs,gate1_frs,gate2_frs] = model(inputs)
        pred1 = torch.argmax(out1, dim=1)
        pred2 = torch.argmax(out2, dim=1)
        num_correct1 += torch.sum(pred1 == target1).item()
        num_data1 += target1.size(0)
        num_correct2 += torch.sum(pred2 == target2).item()
        num_data2 += target2.size(0)
    return num_correct1 / num_data1 , num_correct2 / num_data2


def eval_auc(model, loader, device):
    model.eval()
    outs = []
    labels = []
    for s, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        out = model(inputs)
        out = F.softmax(out, dim=1)
        outs.append(out.detach().cpu().numpy())
        labels.append(targets.detach().cpu().numpy())
    outs = np.concatenate(outs, axis=0)
    labels = np.concatenate(labels, axis=0)
    return roc_auc_score(np.eye(model.out_dim)[labels], outs)


def eval_mse(model, loader, device):
    model.eval()
    tol_error = 0
    num_data = 0
    for s, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        out = model(inputs)
        tol_error += ((out - targets)**2).sum().item()
        num_data += inputs.size(0)
    return tol_error / num_data


def eval_rmse(model, loader, device):
    return eval_mse(model, loader, device) ** 0.5


eval_func = {
    "acc": eval_acc,
    "auc": eval_auc,
    "mse": eval_mse,
    "rmse": eval_rmse,
}



