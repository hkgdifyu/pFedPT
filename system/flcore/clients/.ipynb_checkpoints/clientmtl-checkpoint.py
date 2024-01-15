import torch
import torch.nn as nn
from flcore.clients.clientbase import Client
import numpy as np
import time
import math
import copy


class clientMTL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.omega = None
        self.W_glob = None
        self.idx = 0
        self.itk = args.itk
        self.lamba = 1e-4

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.5)

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model = self.load_model('model')
        # self.model.to(self.device)
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

                self.W_glob[:, self.idx] = flatten(self.model)
                loss_regularizer = 0
                loss_regularizer += self.W_glob.norm() ** 2

                # for i in range(self.W_glob.shape[0] // self.itk):
                #     x = self.W_glob[i * self.itk:(i+1) * self.itk, :]
                #     loss_regularizer += torch.sum(torch.sum((x*self.omega), 1)**2)
                loss_regularizer += torch.sum(torch.sum((self.W_glob*self.omega), 1)**2)
                f = (int)(math.log10(self.W_glob.shape[0])+1) + 1
                loss_regularizer *= 10 ** (-f)

                loss += loss_regularizer
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()
        # self.save_model(self.model, 'model')
        self.omega = None
        self.W_glob = None

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    
    def receive_values(self, W_glob, omega, idx):
        self.omega = torch.sqrt(omega[0][0])
        self.W_glob = copy.deepcopy(W_glob)
        self.idx = idx


def flatten(model):
    state_dict = model.state_dict()
    keys = state_dict.keys()
    W = [state_dict[key].flatten() for key in keys]
    return torch.cat(W)