import torch
import torch.nn as nn
from flcore.clients.clientbase import Client
import numpy as np
import time
import copy


class clientAMP(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.alphaK = args.alphaK
        self.lamda = args.lamda
        self.client_u = copy.deepcopy(self.model)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        self.model.to(self.device)
        self.client_u.to(self.device)
        self.model.train()
        
        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)

                params = weight_flatten(self.model)
                params_ = weight_flatten(self.client_u)
                sub = params - params_
                loss += self.lamda/self.alphaK/2 * torch.dot(sub, sub)

                loss.backward()
                self.optimizer.step()

        self.model.cpu()
        self.client_u.cpu()
        del trainloader

        # print(torch.dot(sub, sub))

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_parameters(self, model, coef_self):
        for new_param, old_param in zip(model.parameters(), self.client_u.parameters()):
            old_param.data = (new_param.data + coef_self * old_param.data).clone()


def weight_flatten(model):
    params = []
    for u in model.parameters():
        params.append(u.view(-1))
    params = torch.cat(params)

    return params
