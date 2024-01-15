import torch
import torch.nn as nn
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data


class clientFomo(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.num_clients = args.num_clients
        self.old_model = copy.deepcopy(self.model)
        self.received_ids = []
        self.received_models = []
        self.weight_vector = torch.zeros(self.num_clients, device=self.device)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.val_ratio = 0.2
        self.train_samples = self.train_samples * (1-self.val_ratio)


    def train(self):
        trainloader, val_loader = self.load_train_data()
        start_time = time.time()

        self.aggregate_parameters(val_loader)
        self.clone_model(self.model, self.old_model)

        self.model.to(self.device)
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
                loss.backward()
                self.optimizer.step()

        self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        val_idx = -int(self.val_ratio*len(train_data))
        val_data = train_data[val_idx:]
        train_data = train_data[:val_idx]

        trainloader = DataLoader(train_data, self.batch_size, drop_last=True, shuffle=True)
        val_loader = DataLoader(val_data, self.batch_size, drop_last=self.has_BatchNorm, shuffle=True)

        return trainloader, val_loader

    def train_metrics(self):
        trainloader, val_loader = self.load_train_data()
        # self.model = self.load_model('model')
        self.model.to(self.device)
        self.model.eval()

        train_num = 0
        loss = 0
        for x, y in trainloader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            output = self.model(x)
            train_num += y.shape[0]
            loss += self.loss(output, y).item() * y.shape[0]

        self.model.cpu()
        # self.save_model(self.model, 'model')

        return loss, train_num
    
    def receive_models(self, ids, models):
        self.received_ids = ids
        self.received_models = models

    def weight_cal(self, val_loader):
        weight_list = []
        L = self.recalculate_loss(self.old_model, val_loader)
        for received_model in self.received_models:
            params_dif = []
            for param_n, param_i in zip(received_model.parameters(), self.old_model.parameters()):
                params_dif.append((param_n - param_i).view(-1))
            params_dif = torch.cat(params_dif)

            weight_list.append((L - self.recalculate_loss(received_model, val_loader)) / (torch.norm(params_dif) + 1e-5))

        # import torch.autograd.profiler as profiler
        # with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        #     self.weight_vector_update(weight_list)
        # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        
        # from pytorch_memlab import LineProfiler
        # with LineProfiler(self.weight_vector_update(weight_list)) as prof:
        #     self.weight_vector_update(weight_list)
        # prof.display()

        self.weight_vector_update(weight_list)

        return torch.tensor(weight_list)
        
    # from pytorch_memlab import profile
    # @profile
    def weight_vector_update(self, weight_list):
        # self.weight_vector = torch.zeros(self.num_clients, device=self.device)
        # for w, id in zip(weight_list, self.received_ids):
        #     self.weight_vector[id] += w.clone()
    
        self.weight_vector = np.zeros(self.num_clients)
        for w, id in zip(weight_list, self.received_ids):
            self.weight_vector[id] += w.item()
        self.weight_vector = torch.tensor(self.weight_vector).to(self.device)

    def recalculate_loss(self, new_model, val_loader):
        L = 0
        new_model.to(self.device)
        for x, y in val_loader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            output = new_model(x)
            loss = self.loss(output, y)
            L += loss.item()
        new_model.cpu()
        return L / len(val_loader)

    def add_parameters(self, w, received_model):
        for param, received_param in zip(self.model.parameters(), received_model.parameters()):
            param.data += received_param.data.clone() * w
        
    def aggregate_parameters(self, val_loader):
        weights = self.weight_scale(self.weight_cal(val_loader))

        if len(weights) > 0:
            for param in self.model.parameters():
                param.data.zero_()

            for w, received_model in zip(weights, self.received_models):
                self.add_parameters(w, received_model)

    def weight_scale(self, weights):
        weights = torch.maximum(weights, torch.tensor(0))
        w_sum = torch.sum(weights)
        if w_sum > 0:
            weights = [w/w_sum for w in weights]
            return torch.tensor(weights)
        else:
            return torch.tensor([])
