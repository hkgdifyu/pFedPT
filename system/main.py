#!/usr/bin/env python
import sys

print(sys.path)
import copy
import torch
import argparse
import os
print (os.path.expandvars('$HOME'))
import time
import warnings
import numpy as np
from torch.nn.functional import dropout
import torchvision
from flcore.servers.serveravg import FedAvg
from flcore.servers.serverpfedpt import PFedPT
from flcore.trainmodel.models import *
from utils.result_utils import average_data
from utils.mem_utils import MemReporter

warnings.simplefilter("ignore")

def run(args):
    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
    

        if model_str == "cnn":
            if args.dataset == "mnist" or args.dataset == "fmnist":
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024)
            elif args.dataset == "Cifar10" or args.dataset == "Cifar100":
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)

            elif args.dataset[:13] == "Tiny-imagenet" or args.dataset[:8] == "Imagenet":
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)

        elif model_str == "vit":
            if args.dataset == "Cifar10":
                args.model = VisionTransformer(num_classes=10)
            elif args.dataset[:13] == "Tiny-imagenet":
                args.model = VisionTransformer(img_size=64,num_classes=200)
            else:
                args.model = VisionTransformer(num_classes=100)

        else:
            raise NotImplementedError

        # select algorithm
        if args.algorithm == "FedAvg":
            server = FedAvg(args, i)

        elif args.algorithm == "PFedPT":
            if args.dataset == "Tiny-imagenet":
                args.generator = copy.deepcopy(PadPrompter(inchanel=1, pad_size=args.num_prompt, image_size=64, args=args))
            else:
                args.generator = copy.deepcopy(PadPrompter(inchanel=3, pad_size=args.num_prompt, image_size=32, args=args))

            args.model = LocalModel_pt(args.generator, args.model)
            server = PFedPT(args, i)


        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    # average_data(dataset=args.dataset,
    #              algorithm=args.algorithm,
    #              goal=args.goal,
    #              times=args.times,
    #              length=args.global_rounds / args.eval_gap + 1)

    print("All done!")

    reporter.report()



if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="Cifar10")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-p', "--predictor", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.05,
                        help="Local learning rate")
    parser.add_argument("--learning_decay", type=float, default=1,
                        help="weight decay")
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='momentum')
    parser.add_argument('-gr', "--global_rounds", type=int, default=30)
    parser.add_argument('-ls', "--local_steps", type=int, default=5)
    parser.add_argument('-algo', "--algorithm", type=str, default="PFedPT")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1,
                        help="Ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=10,
                        help="Total number of clients")
    parser.add_argument('-np', "--num_prompt", type=int, default=2,
                        help="Size of prompt")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='models')

    # read-data
    parser.add_argument("--arv1", type=str, default="noniid")
    parser.add_argument("--arv2", type=str, default="-")
    parser.add_argument("--arv3", type=str, default="pat")
    parser.add_argument("--arv4", type=str, default="50")
    parser.add_argument("--arv5", type=str, default="0.1")
    parser.add_argument("--arv6", type=str, default="5")


    # pfedpt

    parser.add_argument('--pt_learning_rate', type=float, default=20,
                        help='learning rate')

    parser.add_argument('-pls', "--plocal_steps", type=int, default=5)



    # practical
    parser.add_argument('-dep','--depth', type=int, default=6)
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Dropout rate for clients")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_steps))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Time select: {}".format(args.time_select))
    print("Time threthold: {}".format(args.time_threthold))
    print("Global rounds: {}".format(args.global_rounds))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Local model: {}".format(args.model))
    print("Using device: {}".format(args.device))

    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("=" * 50)

    run(args)
