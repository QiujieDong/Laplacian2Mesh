#!/usr/bin/env bash

## run the training
python train_seg.py \
--data_path data/human_seg \
--name humanbody \
--num_classes 8 \
--num_inputs 512 128 32 \
--lr 3e-3 \
--batch_size 128 \
--scheduler_mode Warmup \
--scheduler_T0 50 \
--warm_up_epochs 20 \
--warm_up_T_max 100 \
--scheduler_eta_min 3e-6 \
--weight_decay 0.3 \
--loss_rate 5e-3 \
--bandwidth 1.0 \
--prefetch_factor 2 \
