#!/usr/bin/env bash

## run the training
python train_seg.py \
--data_path data/coseg_chairs \
--name chairs \
--num_classes 3 \
--num_inputs 256 64 16 \
--lr 1e-3 \
--batch_size 256 \
--scheduler_mode CosWarm \
--scheduler_T0 40 \
--scheduler_eta_min 3e-6 \
--weight_decay 0.1 \
--loss_rate 5e-2 \
--bandwidth 0.8 \
--prefetch_factor 2 \
