import os
import argparse

import tqdm
import numpy as np

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchmetrics import Accuracy

from network.Net_cls import Net, init_weights
from dataset.base_dataset import BaseDataset

from util.utils import same_seed
from util.utils import RunningAverage
from util.utils import save_logging

import wandb
import matplotlib

# display error, if we use plt.savefig() in Linux。using matplotlib.use('Agg') to solve the problem.
matplotlib.use('Agg')


# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def train(args, net, train_dl, criterion, writer, epoch):
    net.train()

    preds = []
    labels = []

    train_loss_avg = RunningAverage()
    train_acc_avg = RunningAverage()

    accuracy = Accuracy(num_classes=args.num_classes, ignore_index=None).to(args.device)

    with tqdm.tqdm(total=len(train_dl)) as t:
        for j, (datas, gt) in enumerate(train_dl):
            # load train data
            level_0 = datas['levels'][0].to(args.device)
            level_1 = datas['levels'][1].to(args.device)
            level_2 = datas['levels'][2].to(args.device)
            c_1 = datas['c'][0].to(args.device)
            c_2 = datas['c'][1].to(args.device)
            c_3 = datas['c'][2].to(args.device)
            # load ground truth
            gt = gt.to(args.device)

            pred = net(level_0, level_1, level_2, c_1, c_2, c_3)
            pred_label = F.log_softmax(pred, dim=-1)

            # computing loss
            optimizer.zero_grad()
            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()

            acc = accuracy(pred_label, gt)

            preds.append(pred_label.cpu())
            labels.append(gt.cpu())

            train_loss_avg.update(loss.cpu().item())
            train_acc_avg.update(acc.cpu().item())

            t.set_postfix(loss='{:05.4f}'.format(train_loss_avg()))
            t.update()

            del level_0, level_1, level_2, c_1, c_2, c_3
            torch.cuda.empty_cache()

    # tensorboard log: train
    writer.add_scalar('Accuracy/train', train_acc_avg(), epoch)
    writer.add_scalar('Loss/train', train_loss_avg(), epoch)

    return train_loss_avg(), train_acc_avg()


def test(args, net, test_dl, criterion=None, writer=None, epoch=None):
    net.eval()

    preds = []
    labels = []

    test_loss_avg = RunningAverage()
    test_acc_avg = RunningAverage()

    accuracy = Accuracy(num_classes=args.num_classes, ignore_index=None).to(args.device)

    with torch.no_grad():
        for j, (datas, gt) in enumerate(test_dl):
            # load the test data
            level_0 = datas['levels'][0].to(args.device)
            level_1 = datas['levels'][1].to(args.device)
            level_2 = datas['levels'][2].to(args.device)
            c_1 = datas['c'][0].to(args.device)
            c_2 = datas['c'][1].to(args.device)
            c_3 = datas['c'][2].to(args.device)
            # load ground truth
            gt = gt.to(args.device)

            pred = net(level_0, level_1, level_2, c_1, c_2, c_3)
            pred_label = F.log_softmax(pred, dim=-1)

            acc = accuracy(pred_label, gt)

            if criterion is not None:
                loss = criterion(pred, gt)
                test_loss_avg.update(loss.cpu().item())

            preds.append(pred_label.cpu())
            labels.append(gt.cpu())

            test_acc_avg.update(acc.cpu().item())

            del level_0, level_1, level_2, c_1, c_2, c_3
            torch.cuda.empty_cache()

    if args.mode == 'train':
        # tensorboard log: test
        writer.add_scalar('Accuracy/val', test_acc_avg(), epoch)
        writer.add_scalar('Loss/val', test_loss_avg(), epoch)

    return test_loss_avg(), test_acc_avg()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--seed', type=int, default=40938661)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--scheduler_mode', choices=['CosWarm', 'MultiStep'], default='CosWarm')
    parser.add_argument('--scheduler_T0', type=int, default=30)
    parser.add_argument('--scheduler_eta_min', type=float, default=3e-7)
    parser.add_argument('--weight_decay', type=float, default=0.3)
    parser.add_argument('--amsgrad', action='store_true')
    parser.add_argument('--loss_rate', type=float, default=1.8)
    parser.add_argument('--bandwidth', type=float, default=1.0)
    parser.add_argument('--num_classes', type=int, default=30)
    parser.add_argument('--num_inputs', nargs='+', default=[154, 64, 16], type=int, help='Multi-resolution input')
    parser.add_argument('--data_path', type=str, default='data/Manifold40_NoOrient')
    parser.add_argument('--name', type=str, default='base')

    args = parser.parse_args()
    print(args)

    same_seed(args.seed)

    print('Load Dataset...')
    base_dataset = BaseDataset(args)
    if args.mode == 'train':
        train_dl, test_dl = base_dataset.classification_dataset()
    else:
        test_dl = base_dataset.classification_dataset()
    # define the Net
    net = Net(args.num_classes, args.num_inputs[-1]).to(args.device)

    # save the checkpoints
    if not os.path.exists(os.path.join('checkpoints', args.name)):
        os.makedirs(os.path.join('checkpoints', args.name))

    # save the visualization result
    if args.mode == 'test' and (not os.path.exists(os.path.join('visualization_result', args.name))):
        os.makedirs(os.path.join('visualization_result', args.name))

    min_val_loss = np.inf
    max_val_acc = -np.inf

    if args.mode == 'train':

        # Use wandb to visualize the training process
        wandb.init(project='lap_cls', entity='laplacian2mesh', config=args, name=args.name, sync_tensorboard=True,
                   settings=wandb.Settings(start_method="fork"))
        wandb.watch(net, log="gradients", log_graph=False)
        # tensorboard
        writer = SummaryWriter(os.path.join('checkpoints', args.name, 'log_dir'))

        # Network initialization
        init_weights(net)

        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
        criterion = nn.CrossEntropyLoss().to(args.device)

        # select scheduler mode
        if args.scheduler_mode == 'CosWarm':
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.scheduler_T0, T_mult=2,
                                                                 eta_min=args.scheduler_eta_min, verbose=True)
        else:
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[200, 500, 1000], gamma=0.1, verbose=True)

        for epoch in tqdm.trange(args.epochs):
            train_loss, train_acc = train(args, net, train_dl, criterion, writer, epoch)
            test_loss, test_acc = test(args, net, test_dl, criterion, writer, epoch)

            scheduler.step()
            writer.add_scalar('Utils/lr_scheduler', scheduler.get_last_lr()[0], global_step=epoch)

            save_logging(args, test_loss, test_acc, epoch, net, train_loss, train_acc, save_name='last')
            # save best loss
            if min_val_loss > test_loss:
                min_val_loss = test_loss
                save_logging(args, test_loss, test_acc, epoch, net, train_loss, train_acc, save_name='best_loss')
            # save best acc
            if max_val_acc < test_acc:
                max_val_acc = test_acc
                save_logging(args, test_loss, test_acc, epoch, net, train_loss, train_acc, save_name='best_acc')
        writer.close()
    else:
        net.load_state_dict(torch.load(os.path.join('checkpoints', args.name, 'best_acc.pth')))
        criterion = nn.CrossEntropyLoss().to(args.device)
        test_loss, test_acc = test(args, net, test_dl, criterion)
        save_logging(args, test_loss, test_acc)
