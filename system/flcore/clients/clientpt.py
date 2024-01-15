import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
import copy
from sklearn.preprocessing import label_binarize
from sklearn import metrics

# from utils.privacy import *


class clientPT(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)



        self.plocal_steps = args.plocal_steps
        self.poptimizer = torch.optim.SGD(self.model.generator.parameters(), lr=args.pt_learning_rate,
                                          momentum=args.momentum)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.poptimizer, step_size=self.plocal_steps,
                                                         gamma=args.learning_decay)

    def train(self):

        trainloader = self.load_train_data()

        start_time = time.time()
        old_prompt = copy.deepcopy(self.model.generator)

        self.model.to(self.device)
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
                self.scheduler.step()

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
    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_acc2 = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                output2 = self.model.base(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_acc2 += (torch.sum(torch.argmax(output2, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                y_true.append(label_binarize(y.detach().cpu().numpy(), classes=np.arange(self.num_classes)))

        self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_acc2, test_num, auc