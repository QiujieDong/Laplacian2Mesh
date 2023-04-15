#!/usr/bin/env bash

## run the training
python train_seg.py \
--mode test \
--data_path data/coseg_vases \
--name vases \
--num_classes 4 \
--num_inputs 256 64 16 \
--weight_decay 0.3 \
--loss_rate 1.8 \
--bandwidth 1.0 \
--prefetch_factor 2 \
