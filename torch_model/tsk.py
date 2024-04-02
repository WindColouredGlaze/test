"""
from pytsk.torch_model import TSK
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as Init
import numpy as np
from antecedents import AnteGaussianAndHTSK, AnteGaussianAndLogTSK, AnteGaussianAndTSK, GaussianAntecedent


class Consequent(nn.Module):  # 主网络的后件
    def __init__(self, in_dim, out_dim, n_rules):
        super(Consequent, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_rules = n_rules
        self.cons = nn.ModuleList([nn.Linear(self.in_dim, self.n_rules) for _ in range(self.n_rules)])  # consequents
        # nn.ModuleList，它是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器。

    def forward(self, x, frs):
        # for cons in self.cons:
        #     t = cons.weight.data
        #     print("l2 loss", t)
        cons = torch.cat([cons(x).unsqueeze(1) for cons in self.cons], dim=1)  # 计算后件输出y1-yr拼接矩阵
        # print ("cons:",cons)
        fea_out = torch.sum(frs.unsqueeze(2) * cons, dim=1)  # torch.unsqueeze()这个函数主要是对数据维度进行扩充，unsqueeze(2)表示在第二维增加一个维度
        return fea_out

    def get_consparam(self):
        # wight = torch.cat([cons.weight.data for cons in self.cons], dim=1)
        # print("wight:", wight)
        return torch.cat([cons.weight.data for cons in self.cons], dim=1)



class Fullconnect(nn.Module):
    def __init__(self, in_dim, out_dim, n_rules):
        super(Fullconnect, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim[0]
        self.n_rules = n_rules
        self.cons = nn.Linear(n_rules, out_dim[0])     # 定义网络中的全连接层，将m*n_rules的张量转换为m*out_dim的张量

    def forward(self, fea, frs):
        wightfea = frs * fea
        return self.cons(wightfea)     # 矩阵相乘，计算上面定理的全连接层


class TSK(nn.Module):
    def __init__(self, in_dim, out_dim, n_rules, order=1,
                 antecedent="tsk"):
        """
        :param in_dim: input dimension.
        :param out_dim: output dimension. C for a $C$-class classification problem, 1 for a single output regression
                    problem
        :param n_rules: number of rules.
        :param mf: type of membership function. Support: ["gaussian"]
        :param tnorm: type of t-norm. Support: ["and", "or"]. "and" means using Prod t-norm. "or" means using
                    Min t-norm.
        :param defuzzy: defuzzy type. Support: ["tsk", "htsk", "log"]
                    "tsk": weighted average, $y=\sum_r^R \frac{f_ry_r}{\sum_r^R f_r}$
                    "htsk": htsk defuzzy in [1].
                    "log": Log defuzzy in [1],[2].
        """
        super(TSK, self).__init__()     # 继承父类nn.Module的__init__()方法，并对其中一些参数进行修改，Python3.x 中则是super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_rules = n_rules
        self.antecedent = antecedent
        self.order = order
        self._build_model()

    def _build_model(self):
        self.cons = Consequent(self.in_dim, self.out_dim, self.n_rules)
        self.task1_fullcon = Fullconnect(self.n_rules, self.out_dim, self.n_rules)

        if self.antecedent == "tsk":
            self.firing_level = AnteGaussianAndTSK(self.in_dim, self.n_rules)
            self.gate1_firing_level = AnteGaussianAndTSK(self.in_dim, self.n_rules)
        elif self.antecedent == "htsk":
            self.firing_level = AnteGaussianAndHTSK(self.in_dim, self.n_rules)
            self.gate1_firing_level = AnteGaussianAndHTSK(self.in_dim, self.n_rules)
        elif self.antecedent == "logtsk":
            self.firing_level = AnteGaussianAndLogTSK(self.in_dim, self.n_rules)
            self.gate1_firing_level = AnteGaussianAndLogTSK(self.in_dim, self.n_rules)
        elif callable(self.antecedent):     # callable() 函数用于检查一个对象是否是可调用的
            self.firing_level = self.antecedent(self.in_dim, self.n_rules)
            self.gate1_firing_level = self.antecedent(self.in_dim, self.n_rules)
        else:
            raise ValueError("Unsupported firing level type")

    def init_model(self, X, y=None, scale=1, std=0.2, method="cluster", sigma=None, cluster_kwargs=None, eps=1e-8):
        self.firing_level.init_model(X, y, scale=scale, std=std, method=method, sigma=sigma, cluster_kwargs=cluster_kwargs, eps=eps)
        self.gate1_firing_level.init_model(X, y, scale=scale, std=std, method=method, sigma=sigma, cluster_kwargs=cluster_kwargs, eps=eps)


    def forward(self, X, **kwargs):     # kwargs在args之后表示成对键值对(字典)
        frs = self.firing_level(X)      # 专家模糊网络的激活水平输出
        gate1_frs = self.gate1_firing_level(X)  # Task1门控网络输出的权值
        fea_out = self.cons(X, frs)         # 计算主网络模糊特征
        out1 = self.task1_fullcon(fea_out, gate1_frs)    # 计算全连接层输出1

        if kwargs.pop("frs", False):    # kwargs.pop()用来删除关键字参数中的末尾元素
            return out1,  frs, gate1_frs
        else:
            return out1,  frs, gate1_frs

    def save(self, path=None):
        torch.save(self.state_dict(), path)

    def load(self, path=None):
        self.load_state_dict(torch.load(path))
        self.firing_level.inited = True
        self.gate1_firing_level.inited = True

    def antecedent_params(self, name=True):
        if name:
            return self.firing_level.named_parameters(), self.gate1_firing_level.named_parameters()
        else:
            return self.firing_level.parameters(), self.gate1_firing_level.parameters()

    def predict(self, X):
        my_tensor = next(self.parameters())
        cuda_check = my_tensor.is_cuda
        if cuda_check:
            device = my_tensor.get_device()
        else:
            device = "cpu"

        if isinstance(X, np.ndarray):    # 如果X是其 N 维数组 ndarray 类的对象
            X = torch.as_tensor(X).float().to(device)
            # out = self(X)
            [out1,  frs, gate1_frs] = self(X)
            out = out1
            return out.detach().cpu().numpy() # 将cpu上的tensor转为numpy数据
        else:
            outs = []
            for s, (inputs, _) in enumerate(X):
                inputs = inputs.to(device)
                # out = self(inputs)
                [out1, frs, gate1_frs] = self(inputs)
                out = out1
                outs.append(out.detach().cpu().numpy())
            return np.concatenate(outs, axis=0)     # 数组拼接

    def predict_score(self, X):
        my_tensor = next(self.parameters())
        cuda_check = my_tensor.is_cuda
        if cuda_check:
            device = my_tensor.get_device()
        else:
            device = "cpu"

        if isinstance(X, np.ndarray):    # 如果X是  ndarray 类的N 维数组
            X = torch.as_tensor(X).float().to(device)
            [out1, frs, gate1_frs]= self(X)
            out1 = F.softmax(out1, dim=1)
            return out1.detach().cpu().numpy()
        else:
            outs = []
            for s, (inputs, _) in enumerate(X):
                inputs = inputs.to(device)
                [out1, frs, gate1_frs] = self(inputs)
                out1 = F.softmax(out1, dim=1)
                outs.append(out1.detach().cpu().numpy())
            return np.concatenate(outs, axis=0)

    def l2_loss(self):
        """
        compute the l2 loss using the consequent parameters except the bias
        :return:
        """
        w = self.cons.get_consparam()
        l2loss = torch.sum(w ** 2)
        # print("l2 loss", l2loss)
        return l2loss

    def ur_loss(self, frs, n_classes):
        """
        UR loss in our paper
        :param frs: normalized firing level for one batch
        :param n_classes: Number of classes for tasks
        :return:
        """
        return ((torch.mean(frs, dim=0) - 1/n_classes)**2).sum()