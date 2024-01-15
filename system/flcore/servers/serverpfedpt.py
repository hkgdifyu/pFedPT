from flcore.clients.clientpt import clientPT
from flcore.servers.serverbase import Server
from threading import Thread
import time

import torch
import os
import h5py
import copy
import numpy as np
class PFedPT(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        

        # select slow clients
        self.set_slow_clients()
        self.args = args
        self.set_clients(args, clientPT)
        self.global_model = copy.deepcopy(args.model.base)
        self.diff_pro = []
        self.clients_diverge = []

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        local_acc = []
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
            temp_diff_pro = 0

            for client in self.selected_clients:
                temp_diff_pro_client = client.train()
                temp_diff_pro = temp_diff_pro +temp_diff_pro_client.item()
            print("Averaged prompr difference: {:.4f}".format(temp_diff_pro))
            self.diff_pro.append(temp_diff_pro)
            diverge_clents =0
            for new_param, old_param in zip(self.clients[0].model.generator.parameters(), self.clients[1].model.generator.parameters()):
                diff_pro = new_param - old_param
                diff_pro = torch.where(diff_pro > 0, diff_pro, torch.zeros_like(diff_pro) - diff_pro)
                diverge_clents = diverge_clents+torch.sum(diff_pro)
            print("0 and 1 clients difference: {:.4f}".format(diverge_clents.item()))
            self.clients_diverge.append(diverge_clents.item())
            if i%self.eval_gap == 0:
                print("\nEvaluate local model")
                self.evaluate(acc=local_acc)
            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        print("\nBest local accuracy.")
        print(max(local_acc))
        self.save_results()
        self.save_global_model()
        self.save_client_model()

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(copy.deepcopy(client.model.base))
    def save_client_model(self):
        model_path = os.path.join("models", self.dataset,"client",self.args.arv1+"*"+self.args.arv2+"*"+self.args.arv3+"*"+self.args.arv4+"*"+self.args.arv5+"*"+self.args.arv6 )
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        for c_idx,c in enumerate(self.clients):
            model_path_save = os.path.join(model_path, self.algorithm + "_client" +str(c_idx)+ "_" + str(self.args.num_prompt) + "_" + str(self.args.join_ratio) + "_" + str(self.args.num_clients)+ "_" + str(self.args.plocal_steps) + "_" + str(self.args.global_rounds)+ ".pt")
            torch.save(c.model, model_path_save)
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"+ self.args.arv1+"*"+self.args.arv2+"*"+self.args.arv3+"*"+self.args.arv4+"*"+self.args.arv5+"*"+self.args.arv6 +"/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo =algo + "_" + self.goal + "_" + str(self.times)+ "_" + str(self.args.num_prompt) + "_" + str(self.args.join_ratio) + "_" + str(self.args.num_clients)+ "_" + str(self.args.plocal_steps) + "_" + str(self.args.global_rounds)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_acc_std', data=self.rs_test_acc_std)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('diff_pro', data=self.diff_pro)
                hf.create_dataset('clients_diverge', data=self.clients_diverge)

    def evaluate(self, acc=None, loss=None):

        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_acc2 = sum(stats[4]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        self.rs_test_auc.append(test_auc)
        self.rs_test_acc_std.append(np.std(accs))
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test oral Accurancy: {:.4f}".format(test_acc2))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_correct2 = []
        tot_auc = []

        for c in self.clients:
            ct, ct2, ns, auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_correct2.append(ct2 * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc, tot_correct2