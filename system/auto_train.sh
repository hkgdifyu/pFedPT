#!/bin/bash


# ===============================================================horizontal(Cifar10-iid)======================================================================


 rm ../dataset/Cifar10/config.json
 cd ../dataset/
 nohup python -u generate_cifar10.py noniid - dir 50 0.3 10 > cifar10_dir0.3_dataset.out 2>&1
 cd ../system/



nohup python -u main.py  --arv1 "noniid" --arv2 "-" --arv3 "dir" --arv4  "50" --arv5  "0.3" --arv6  "-"  --device_id  0 -algo PFedPT  -m cnn -lbs 16 -nc 50 -jr 0.2   -pls 5 -gr 150 -ls 5 -np 2 -data "Cifar10" -nb 10 --pt_learning_rate 1 > log.txt 2>&1 &


