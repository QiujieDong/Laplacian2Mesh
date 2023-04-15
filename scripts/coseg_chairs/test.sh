#!/usr/bin/env bash

## run the testing
python train_seg.py \
--mode test \
--data_path data/coseg_chairs \
--name chairs \
--num_classes 3 \
--num_inputs 256 64 16 \
--weight_decay 0.1 \
--loss_rate 0.05 \
--bandwidth 0.8 \
--prefetch_factor 2 \
