import torch
from collections import OrderedDict
from sklearn.utils.multiclass import type_of_target
from .loader import classification_loader, continuous_loader


def check_grad(model):
    """
    check if gradients of each model contain nan
    :param model: pytorch model
    :return: bool, True if gradients contain nan, else False
    """
    for n, p in model.named_parameters():
        #print(n,p)
        # print("p.grad:\n")
        # print(n,p.grad)
        # print(p.grad)
        if torch.sum(torch.isnan(p.grad)) > 0:
            print("Warning: Gradient contains nan, skipping this batch...")
            return True
    return False


class Trainer:
    def __init__(self, model, n_rules, outdim, optimizer, criterion, device="cuda", callbacks=None, verbose=0):
        """

        :param model: pytorch model
        :param outdim: Number of classes of output
        :param optimizer: pytorch optimizer
        :param criterion: loss function
        :param device: device, pytorch device format
        :param callbacks: callback during the training
        :param verbose: > 0, show the log, =0 disable log
        """
        self.model = model
        self.n_rules = n_rules
        self.outdim = outdim
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.callbacks = [] if callbacks is None else callbacks
        self.model.to(self.device)
        self.verbose = verbose
        self.batch_size = None
        assert isinstance(self.callbacks, list), "callbacks must be an list, but got: {}".format(callbacks)

    def fit(self, X, y, max_epoch=1, w_l2=0, w_ur=0, batch_size=32, shuffle=False, forward_args=None, clip_sigma=True, loader_type="auto"):
        self.batch_size = batch_size
        if loader_type == "auto":
            target_type = type_of_target(y)
            if "continuous" in target_type:
                train_loader = continuous_loader(X, y, batch_size=batch_size, shuffle=shuffle)
            else:
                train_loader = classification_loader(X, y, batch_size=batch_size, shuffle=shuffle)
        elif loader_type == "classification":
            train_loader = classification_loader(X, y, batch_size=batch_size, shuffle=shuffle)
        elif loader_type == "continuous":
            train_loader = continuous_loader(X, y, batch_size=batch_size, shuffle=shuffle)
        else:
            raise ValueError("Wrong loader_type, only support: [auto, continuous, classification]")
        return self.fit_loader(train_loader, max_epoch, w_l2, w_ur, forward_args, clip_sigma)

    def fit_loader(self, train_loader, max_epoch=1, w_l2=0, w_ur=0, forward_args=None, clip_sigma=True):
        if forward_args is None:
            forward_args = dict()

        # register call back
        for callback in self.callbacks:
            callback.register(self)

        self.end_training = False
        self.logs = OrderedDict()

        for callback in self.callbacks:
            callback.on_train_begin(self.logs)

        for e in range(max_epoch):
            self.logs["epoch"] = e
            tol_loss = 0
            for callback in self.callbacks:
                callback.on_epoch_begin(self.logs)
            self.model.train()
            valid_batch = 0
            for b, (inputs, targets) in enumerate(train_loader):
                for callback in self.callbacks:
                    callback.on_batch_begin(self.logs)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                [out1, out2, frs, gate1_frs, gate2_frs]= self.model(inputs, **forward_args)
                # [firing_level, gate1_firing_level, gate2_firing_level] = self.model.antecedent_params()
                # print("targets", targets)
                # print("out1", out1)
                # print("out2", out2)
                # print(frs)
                # print(gate1_frs)
                # loss = 0.5 * self.criterion(out1, targets[:, 0]) + 0.5 * self.criterion(out2, targets[:, 1])
                # loss = 0.5 * self.criterion(out1, targets[:, 0]) + 0.5 * self.criterion(out2, targets[:, 1]) + \
                #        w_l2 * self.model.l2_loss() + \
                #        w_ur * self.model.ur_loss(gate1_frs, self.outdim[0]) + \
                #        w_ur * self.model.ur_loss(gate2_frs, self.outdim[1])
                loss = 0.5 * self.criterion(out1, targets[:, 0]) + 0.5 * self.criterion(out2, targets[:, 1]) + \
                       w_l2 * self.model.l2_loss() + \
                       w_ur * self.model.ur_loss(gate1_frs, self.n_rules) + \
                       w_ur * self.model.ur_loss(gate2_frs, self.n_rules)
                # loss = 0.5 * self.criterion(out1, targets[:, 0]) + 0.5 * self.criterion(out2, targets[:, 1]) + \
                #        w_l2 * self.model.l2_loss() + \
                #        w_ur * self.model.ur_loss(frs, self.n_rules)

                tol_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                if check_grad(self.model):
                    continue
                valid_batch += 1
                self.optimizer.step()
                if clip_sigma:
                    self.model.firing_level.clip_sigma()

                for callback in self.callbacks:
                    callback.on_batch_end(self.logs)
            if valid_batch == 0:
                print("Warning: all batches in this epoch generate nan gradient...")
            self.logs["Loss"] = tol_loss / valid_batch if valid_batch > 0 else float('nan')

            for callback in self.callbacks:
                callback.on_epoch_end(self.logs)

            if self.verbose > 0:
                print_info = ", ".join([
                    "{}: {:.4f}".format(k, v) if isinstance(v, float) else "{}: {}".format(k, v) for k, v in self.logs.items() if k != "epoch"
                ])
                print("[EPOCH {:{width}d}] {}".format(e, print_info, width=len(str(max_epoch))))
            if self.end_training:
                break

        for callback in self.callbacks:
            callback.on_train_end(self.logs)
        return self

    def get_model(self):
        return self.model

    def fit_generator(self, generator, forward_args=None):
        if forward_args is None:
            forward_args = dict()
        for b, (inputs, targets) in generator:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outs = self.model(inputs, **forward_args)
            loss = self.criterion(outs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
