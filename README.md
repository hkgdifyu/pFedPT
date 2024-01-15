#Visual Prompt Based Personalized Federated Learning(TMLR 2024)
This repository contains the official code for our proposed method,This paper has been accepted at TMLR 2021.

![Local Image](.\pipeline.png)

## Environments
With the installed [conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh), we can run this platform in a conda virtual environment called *fl_torch*. Note: due to the code updates, some modules are required to install based on the given `*.yaml`. 
```
conda env create -f env_linux.yaml # for Linux
```



## Datasets 
I currently using **three** famous datasets:  **Cifar10**, **Cifar100** and **Tiny-ImageNet** (Please download the original data from http://cs231n.stanford.edu/tiny-imagenet-200.zip).

they can be easy split into **IID** and **Non-IID** version with `./dataset/utils/dataset_utils.py`. It is easy to add other datasets to this FL platform.

In **Non-IID** setting, two situations exist. The first one is the **pathological Non-IID** setting, the second one is **practical Non-IID** setting. In the **pathological Non-IID** setting, for example, the data on each client only contains the specific number of labels (maybe only two labels), though the data on all clients contains 10 labels such as MNIST dataset. In the **practical Non-IID** setting, Dirichlet distribution is utilized (please refer to this [paper](https://proceedings.neurips.cc/paper/2020/hash/18df51b97ccd68128e994804f3eccc87-Abstract.html) for details). We can input *balance* for the iid setting, where the data are uniformly distributed. 

### Examples for **Cifar10**
- Cifar10
    ```
    cd ./dataset
    python -u generate_cifar10.py iid - - 50 1 10 # for iid and balanced setting
    # python -u generate_cifar10.py noniid - dir 50 0.3 10 # for practical noniid and unbalanced setting with 50 clients and dir $\alpha$ is 0.3
    # python generate_mnist.py noniid - pat 50 1 5 # for pathological noniid and unbalanced setting with 50 clients and each client possesses 5 classes
    ```

## How to start simulating 
- Build dataset: [Datasets](#datasets-updating)

- Train and evaluate the model:
    ```
    cd ./system
    python -u main.py  --arv1 "iid" --arv2 "-" --arv3 "-" --arv4  "50" --arv5  "1" --arv6  "10"  --device_id  0 -algo PFedPT  -m cnn -lbs 16 -nc 50 -jr 0.2   -pls 5 -gr 150 -ls 5 -np 2 -data "Cifar10" -nb 10 --pt_learning_rate 1  # for pFedPT and Cifar10
    ```
    Or you can uncomment the lines you need in `./system/auto_train.sh` and run:
    ```
    cd ./system
    sh auto_train.sh
    ```

**Note**: The hyper-parameters have not been tuned for the algorithms. The values in `./system/auto_train.sh` are just examples. You need to tune the hyper-parameters by yourself. 

# Acknowledgements

Much of the code in this repository was adapted from code in this repository  [PFLlib](https://github.com/TsingZ0/PFLlib), Please refer to their repository for more detailed information.. Their work has greatly contributed to my research.



