import torch
import torch.nn as nn
import numpy as np
import time
from system.flcore.clients.clientbase import Client
import copy


# from utils.privacy import *


class clientT(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.loss = nn.CrossEntropyLoss()
        self.poptimizer = torch.optim.SGD(self.model.generator.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)
        self.ser_para()


        self.plocal_steps = args.plocal_steps

    def ser_para(self):
        torch.manual_seed(self.id)
        self.model.generator.pad_down.data = torch.rand_like(self.model.generator.pad_down.data )
        self.model.generator.pad_left.data = torch.rand_like(self.model.generator.pad_left.data)
        self.model.generator.pad_right.data = torch.rand_like(self.model.generator.pad_right.data)
        self.model.generator.pad_up.data = torch.rand_like(self.model.generator.pad_up.data)

    def train(self):

        trainloader = self.load_train_data()

        start_time = time.time()
        old_prompt = copy.deepcopy(self.model.generator)

        self.model.to(self.device)
        self.model.train()



        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.generator.parameters():
            param.requires_grad = False

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
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
                loss.backward()
                self.optimizer.step()

        self.model.cpu()
        new_prompt = copy.deepcopy(self.model.generator)
        diff_provalue = 0

        for new_param, old_param in zip(old_prompt.parameters(), new_prompt.parameters()):
            diff_pro = new_param - old_param
            diff_pro = torch.where(diff_pro > 0, diff_pro, torch.zeros_like(diff_pro)-diff_pro)
            diff_provalue = diff_provalue + torch.sum(diff_pro)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        return diff_provalue

    def set_parameters(self, base):

        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()