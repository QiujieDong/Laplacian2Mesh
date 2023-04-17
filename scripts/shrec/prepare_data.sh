#!/usr/bin/env bash

python ./prepare/pre_cls_dataset.py \
--data_path data/shrec_16 \
--device cuda \
--augment_orient \
