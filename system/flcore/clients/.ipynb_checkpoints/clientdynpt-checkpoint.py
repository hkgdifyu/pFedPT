import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client


# from utils.privacy import *


class clientDynPT(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        # differential privacy
        # if self.privacy:
        #     check_dp(self.model)
        #     initialize_dp(self.model, self.optimizer, self.sample_rate, self.dp_sigma)

        self.alpha = args.alpha

        self.global_model_vector = None
        old_grad = copy.deepcopy(self.model)
        old_grad = model_parameter_vector(old_grad)
        self.old_grad = torch.zeros_like(old_grad)
        self.poptimizer = torch.optim.SGD(self.model.generator.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)
        self.plocal_steps = args.plocal_steps

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()
        old_prompt = copy.deepcopy(self.model.generator)
        self.model.to(self.device)
        self.old_grad = self.old_grad.to(self.device)
        self.global_model_vector = self.global_model_vector.to(self.device)
        self.model.train()
        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.generator.parameters():
            param.requires_grad = True

        for step in range(self.plocal_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.poptimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.poptimizer.step()
        max_local_steps = self.local_steps
        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.generator.parameters():
            param.requires_grad = False
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

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

                if self.global_model_vector != None:
                    v1 = model_parameter_vector(self.model)
                    loss += self.alpha / 2 * torch.norm(v1 - self.global_model_vector, 2)
                    loss -= torch.dot(v1, self.old_grad)

                loss.backward()

                self.optimizer.step()

        if self.global_model_vector != None:
            v1 = model_parameter_vector(self.model).detach()
            self.old_grad = self.old_grad - self.alpha * (v1 - self.global_model_vector)

        self.model.cpu()
        self.old_grad = self.old_grad.cpu()
        self.global_model_vector = self.global_model_vector.cpu()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        new_prompt = copy.deepcopy(self.model.generator)
        diff_provalue = 0

        for new_param, old_param in zip(old_prompt.parameters(), new_prompt.parameters()):
            diff_pro = new_param - old_param
            diff_pro = torch.where(diff_pro > 0, diff_pro, torch.zeros_like(diff_pro) - diff_pro)
            diff_provalue = torch.sum(diff_pro)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        return diff_provalue
        #
        # if self.privacy:
        #     res, DELTA = get_dp_params(self.optimizer)
        #     print(f"Client {self.id}", f"(ε = {res[0]:.2f}, δ = {DELTA}) for α = {res[1]}")

    def set_parameters(self, model):
        for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

        self.global_model_vector = model_parameter_vector(model).detach().clone()



def model_parameter_vector(model):
    param = [p.view(-1) for p in model.parameters()]
    return torch.cat(param, dim=0)