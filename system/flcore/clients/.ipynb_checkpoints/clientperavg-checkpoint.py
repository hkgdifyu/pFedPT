import numpy as np
import torch
import time
import copy
import torch.nn as nn
from flcore.optimizers.fedoptimizer import PerAvgOptimizer
from flcore.clients.clientbase import Client


class clientPerAvg(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # self.beta = args.beta
        self.beta = self.learning_rate

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = PerAvgOptimizer(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        trainloader = self.load_train_data(self.batch_size*2)
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):  # local update
            for X, Y in trainloader:
                temp_model = copy.deepcopy(list(self.model.parameters()))

                # step 1
                if type(X) == type([]):
                    x = [None, None]
                    x[0] = X[0][:self.batch_size].to(self.device)
                    x[1] = X[1][:self.batch_size]
                else:
                    x = X[:self.batch_size].to(self.device)
                y = Y[:self.batch_size].to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

                # step 2
                if type(X) == type([]):
                    x = [None, None]
                    x[0] = X[0][self.batch_size:].to(self.device)
                    x[1] = X[1][self.batch_size:]
                else:
                    x = X[self.batch_size:].to(self.device)
                y = Y[self.batch_size:].to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()

                # restore the model parameters to the one before first update
                for old_param, new_param in zip(self.model.parameters(), temp_model):
                    old_param.data = new_param.data.clone()

                self.optimizer.step(beta=self.beta)

        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def train_one_step(self):
        testloader = self.load_test_data(self.batch_size)
        iter_testloader = iter(testloader)
        # self.model.to(self.device)
        self.model.train()

        # step 1
        (x, y) = next(iter_testloader)
        if type(x) == type([]):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        y = y.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.loss(output, y)
        loss.backward()
        self.optimizer.step()

        # step 2
        (x, y) = next(iter_testloader)
        if type(x) == type([]):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        y = y.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.loss(output, y)
        loss.backward()
        self.optimizer.step(beta=self.beta)

        # self.model.cpu()
