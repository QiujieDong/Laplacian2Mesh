#!/usr/bin/env bash

python ./prepare/pre_seg_dataset.py \
--data_path data/coseg_vases \
--device cuda \
--augment_orient \
