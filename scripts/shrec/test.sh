## run the training
python train_cls.py \
--mode test \
--data_path data/shrec_16 \
--name shrec_16 \
--num_classes 30 \
--num_inputs 250 64 16 \
--lr 3e-3 \
--batch_size 128 \
--scheduler_mode CosWarm \
--scheduler_T0 50 \
--scheduler_eta_min 3e-7 \
--weight_decay 0.3 \
--loss_rate 5e-3 \
--bandwidth 1.0 \
--prefetch_factor 2 \
--amsgrad \
