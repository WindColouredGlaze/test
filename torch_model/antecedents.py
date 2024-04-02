import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as Init
from sklearn.cluster import KMeans
import numpy as np


support_init_method = ["cluster", "random", "assign"]


class GaussianAntecedent(nn.Module):
    def __init__(self, in_dim, n_rules):
        """
        Antecedent with Gaussian MF
        :param in_dim: int, input dimension
        :param n_rules: int, number of rules
        """
        super(GaussianAntecedent, self).__init__()
        self.in_dim = in_dim
        self.n_rules = n_rules
        self.h = 0.5
        self.inited = False
        self.eps = 1e-8

        self.centers = nn.Parameter(torch.zeros([in_dim, n_rules]), requires_grad=True)
        self.sigmas = nn.Parameter(torch.zeros([in_dim, n_rules]), requires_grad=True)
        # self.centers_gate1 = nn.Parameter(torch.zeros([in_dim, n_rules]), requires_grad=True)
        # self.sigmas_gate1 = nn.Parameter(torch.zeros([in_dim, n_rules]), requires_grad=True)

    def init_model(self, x, y=None, scale=1, std=0.2, method="cluster", sigma=None, cluster_kwargs=None, eps=1e-8):
        self.eps = eps  # allow user to reset eps
        assert method in support_init_method, "wrong init method, only support: {}".format(support_init_method)
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if method == "cluster":
            cluster_kwargs = cluster_kwargs if cluster_kwargs is not None else {}
            km = KMeans(n_clusters=self.n_rules, **cluster_kwargs)
            km.fit(x)
            cluster_center = torch.as_tensor(km.cluster_centers_.T, dtype=torch.float32)
        elif method == "random":
            random_index = np.random.choice(np.arange(x.shape[0]), size=[self.n_rules])
            cluster_center = torch.as_tensor(x[random_index, :].T, dtype=torch.float32)
        elif method == "assign":
            cluster_center = torch.as_tensor(x.T, dtype=torch.float32)
        else:
            raise ValueError("wrong init method")
        self.centers.data = cluster_center
        # self.centers_gate1.data = cluster_center
        if sigma is not None and method == "assign":
            self.sigmas.data[...] = torch.as_tensor(sigma.T, dtype=torch.float32)
            # self.sigmas_gate1.data[...] = torch.as_tensor(sigma.T, dtype=torch.float32)
        else:
            Init.normal_(self.sigmas, mean=scale, std=std)
        self.inited = True

    def get_params(self):
        raise NotImplemented("get_params not implemented")

    def _check_init(self):
        assert self.inited, "model must be inited, try to run:\t model.init_model()"

    def clip_sigma(self, min_eps=1e-8):
        self.sigmas.data[...] = torch.sign(self.sigmas.data) * torch.clamp(self.sigmas.data.abs(), min=min_eps)
        # self.sigmas_gate1.data[...] = torch.sign(self.sigmas_gate1.data) * torch.clamp(self.sigmas_gate1.data.abs(), min=min_eps)
        
    def get_centers(self, return_np=True):
        all_centers = []
        # all_centers_gate1 = []
        for i in range(self.n_rules):
            if return_np:
                all_centers.append(self.centers[:, i].detach().cpu().numpy())
                # all_centers_gate1.append(self.centers_gate1[:, i].detach().cpu().numpy())
            else:
                all_centers.append(self.centers[:, i])
                # all_centers_gate1.append(self.centers_gate1[:, i])
        return all_centers
        # , all_centers_gate1


class AnteGaussianAndTSK(GaussianAntecedent):
    def __init__(self, in_dim, n_rules):
        """
        Antecedent with Gaussian MF, using weighted average defuzzification
        :param in_dim:
        :param n_rules:
        """
        super(AnteGaussianAndTSK, self).__init__(in_dim, n_rules)

    def forward(self, x):
        self._check_init()
        inputs = torch.sum(
            -(x.unsqueeze(dim=2) - self.centers) ** 2 * (self.h / self.sigmas ** 2 + self.eps), dim=1
        )
        frs = F.softmax(inputs, dim=1)
        return frs

    def get_params(self):
        return [self.centers, self.sigmas]

# class Gate1_AnteGaussianAndTSK(GaussianAntecedent):
#     def __init__(self, in_dim, n_rules):
#         """
#         Antecedent with Gaussian MF, using weighted average defuzzification
#         :param in_dim:
#         :param n_rules:
#         """
#         super(Gate1_AnteGaussianAndTSK, self).__init__(in_dim, n_rules)

#     def forward(self, x):
#         self._check_init()
#         inputs = torch.sum(
#             -(x.unsqueeze(dim=2) - self.centers_gate1) ** 2 * (self.h / self.sigmas_gate1 ** 2 + self.eps), dim=1
#         )
#         frs = F.softmax(inputs, dim=1)
#         return frs

#     def get_params(self):
#         return [self.centers_gate1, self.sigmas_gate1]


class AnteGaussianAndHTSK(GaussianAntecedent):
    def __init__(self, in_dim, n_rules):
        """
        Antecedent with Gaussian MF, using HTSK defuzzification [1]
        [1] ....
        :param in_dim: int, input dimension
        :param n_rules: int, number of rules
        """
        super(AnteGaussianAndHTSK, self).__init__(in_dim, n_rules)

    def forward(self, x):
        self._check_init()
        inputs = torch.mean(
            -(x.unsqueeze(dim=2) - self.centers) ** 2 * (self.h / self.sigmas ** 2 + self.eps), dim=1
        )
        frs = F.softmax(inputs, dim=1)
        return frs

    def get_params(self):
        return [self.centers, self.sigmas]


class AnteGaussianAndLogTSK(GaussianAntecedent):
    def __init__(self, in_dim, n_rules):
        """
        Antecedent with Gaussian MF, using LogTSK defuzzification [1, 2]
        [1] ....
        [2] ....
        :param in_dim: int, input dimension
        :param n_rules: int, number of rules
        :param in_dim:
        :param n_rules:
        """
        super(AnteGaussianAndLogTSK, self).__init__(in_dim, n_rules)

    def forward(self, x):
        self._check_init()
        inputs = torch.sum(
            (x.unsqueeze(dim=2) - self.centers) ** 2 * (self.h / self.sigmas ** 2 + self.eps), dim=1
        )
        frs = 1 / inputs
        return frs / torch.sum(frs, dim=1, keepdim=True)

    def get_params(self):
        return [self.centers, self.sigmas]