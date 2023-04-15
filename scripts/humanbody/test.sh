#!/usr/bin/env bash

## run the training
python train_seg.py \
--mode test \
--data_path data/human_seg \
--name humanbody \
--num_classes 8 \
--num_inputs 512 128 32 \
--weight_decay 0.3 \
--loss_rate 5e-3 \
--prefetch_factor 2 \
