#!/bin/bash

echo "==== First round starts ===="

nohup python train.py --logs_dir log --method my --KB_len 20 --distillation_weight 1.0 --distil_para 10.0 --rand 0.1 --lr 1e-1 --num_epochs 4 --device cuda:2 --output xprivacy01.txt > xprivacy1.log 2>&1 &
nohup python train.py --logs_dir log --method my --KB_len 20 --distillation_weight 1.0 --distil_para 10.0 --rand 0.1 --lr 1e-1 --num_epochs 4 --device cuda:1 --output xprivacy02.txt > xprivacy2.log 2>&1 &
nohup python train.py --logs_dir log --method my --KB_len 20 --distillation_weight 1.0 --distil_para 10.0 --rand 0.1 --lr 1e-1 --num_epochs 4 --device cuda:0 --output xprivacy03.txt > xprivacy3.log 2>&1 &


