#!/usr/bin/env bash

## run the training
python train_cls.py \
--data_path data/Manifold40 \
--name manifold40 \
--num_classes 40 \
--num_inputs 154 64 16 \
--lr 3e-4 \
--batch_size 512 \
--scheduler_mode CosWarm \
--scheduler_T0 50 \
--scheduler_eta_min 3e-7 \
--weight_decay 0.3 \
--loss_rate 5e-3 \
--bandwidth 1.0 \
--prefetch_factor 2 \
