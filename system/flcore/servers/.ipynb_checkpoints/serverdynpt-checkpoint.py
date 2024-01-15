import copy
import torch
from system.flcore.clients.clientdynpt import clientDynPT
from system.flcore.servers.serverbase import Server
from threading import Thread
import time
import h5py
import copy
import os

class FedDynPT(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientDynPT)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.diff_pro = []
        self.alpha = args.alpha
        self.clients_diverge = []

        self.server_state = copy.deepcopy(args.model)
        for param in self.server_state.parameters():
            param.data = torch.zeros_like(param.data)

    def train(self):
        local_acc = []
        self.done = False
        i = 0
        # while not self.done:
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
            temp_diff_pro = 0
            for client in self.selected_clients:
                temp_diff_pro_client = client.train()
                temp_diff_pro = temp_diff_pro + temp_diff_pro_client.item()
            print("Averaged prompr difference: {:.4f}".format(temp_diff_pro))
            self.diff_pro.append(temp_diff_pro)
            diverge_clents =0
            for new_param, old_param in zip(self.clients[0].model.generator.parameters(), self.clients[1].model.generator.parameters()):
                diff_pro = new_param - old_param
                diff_pro = torch.where(diff_pro > 0, diff_pro, torch.zeros_like(diff_pro) - diff_pro)
                diverge_clents = diverge_clents+torch.sum(diff_pro)
            print("0 and 1 clients difference: {:.4f}".format(diverge_clents.item()))
            self.clients_diverge.append(diverge_clents.item())

            if i % self.eval_gap == 0:
                print("\nEvaluate local model")
                self.evaluate(acc=local_acc)

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.update_server_state()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

            self.done = self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt)
            i += 1

        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nBest local accuracy.")
        print(max(local_acc))
        print("\nAveraged time per iteration.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

    def add_parameters(self, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() / self.join_clients

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)

        for client_model in self.uploaded_models:
            self.add_parameters(client_model)

        for server_param, state_param in zip(self.global_model.parameters(), self.server_state.parameters()):
            server_param.data -= (1 / self.alpha) * state_param

    def update_server_state(self):
        assert (len(self.uploaded_models) > 0)

        model_delta = copy.deepcopy(self.uploaded_models[0])
        for param in model_delta.parameters():
            param.data = torch.zeros_like(param.data)

        for client_model in self.uploaded_models:
            for server_param, client_param, delta_param in zip(self.global_model.parameters(),
                                                               client_model.parameters(), model_delta.parameters()):
                delta_param.data += (client_param - server_param) / self.num_clients

        for state_param, delta_param in zip(self.server_state.parameters(), model_delta.parameters()):
            state_param.data -= self.alpha * delta_param
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
            self.uploaded_models.append(copy.deepcopy(client.model))
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_acc_std', data=self.rs_test_acc_std)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('diff_pro', data=self.diff_pro)