import time
import torch
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientAVG)

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

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]
            if i%self.eval_gap == 0:
                print("\nEvaluate local model")
                self.evaluate(acc=local_acc)
            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])
        # i= i+1
        #
        # s_t = time.time()
        # self.selected_clients = self.clients
        # self.send_models()
        #
        # if i % self.eval_gap == 0:
        #     print(f"\n-------------Round number: {i}-------------")
        #     print("\nEvaluate global model")
        #     self.evaluate()
        #
        # for client in self.selected_clients:
        #     client.local_steps = 50
        #     client.train()
        #
        # # threads = [Thread(target=client.train)
        # #            for client in self.selected_clients]
        # # [t.start() for t in threads]
        # # [t.join() for t in threads]
        # if i % self.eval_gap == 0:
        #     print("\nEvaluate local model")
        #     self.evaluate(acc=local_acc)
        # self.receive_models()
        # self.aggregate_parameters()
        #
        # self.Budget.append(time.time() - s_t)
        # print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])




        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nBest local accuracy.")
        print(max(local_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()
