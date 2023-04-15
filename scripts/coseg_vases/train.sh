#!/usr/bin/env bash

## run the training
python train_seg.py \
--data_path data/coseg_vases \
--name vases \
--num_classes 4 \
--num_inputs 256 64 16 \
--lr 3e-3 \
--batch_size 256 \
--scheduler_mode Warmup \
--scheduler_T0 40 \
--warm_up_epochs 20 \
--warm_up_T_max 100 \
--scheduler_eta_min 3e-6 \
--weight_decay 0.3 \
--loss_rate 1.8 \
--bandwidth 1.0 \
--prefetch_factor 2 \
