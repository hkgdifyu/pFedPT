import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)
num_clients = int(sys.argv[4])
num_classes = 10
dir_path = "Cifar10/"+sys.argv[1]+"*"+sys.argv[2]+"*"+sys.argv[3]+"*"+sys.argv[4]+"*"+sys.argv[5]+"*"+sys.argv[6]+"/"
dir_path2 = "Cifar10/"

# Allocate data to users
def generate_cifar10(dir_path, num_clients, num_classes, niid, balance, partition,class_per_client,alpha):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition,alpha):
        return
        
    # Get Cifar10 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform = transforms.Compose(
    #     [transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(
        root=dir_path2+"rawdata", train=True, download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(
        root=dir_path2+"rawdata", train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=class_per_client,alpha = alpha)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition,alpha)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    alpha = float(sys.argv[5])
    class_per_client = int(sys.argv[6])
    # niid = True
    # balance = True
    # partition = 'dir'
    generate_cifar10(dir_path, num_clients, num_classes, niid, balance, partition,class_per_client,alpha)