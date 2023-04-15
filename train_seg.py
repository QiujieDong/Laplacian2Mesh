import os
import argparse

import tqdm
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import torch.optim as optim
from torch.optim import lr_scheduler
from torchmetrics import Accuracy

from network.Net_seg import Net, init_weights
from dataset.base_dataset import BaseDataset

from util.utils import same_seed
from util.utils import RunningAverage
from util.utils import save_logging
from util.utils import segmentation_loss
from util.utils import warm_up_with_cosine_lr
from util.utils import faces_label

import wandb
import matplotlib

# display error, if we use plt.savefig() in Linux。using matplotlib.use('Agg') to solve the problem.
matplotlib.use('Agg')


# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def train(args, net, train_dl, weight_train, writer, epoch):

    net.train()

    preds = []
    labels = []

    train_loss_avg = RunningAverage()
    train_acc_avg = RunningAverage()

    accuracy = Accuracy()

    with tqdm.tqdm(total=len(train_dl)) as t:

        for j, (datas, gt) in enumerate(train_dl):
            # load train data
            level_0 = datas['levels'][0].to(args.device)
            level_1 = datas['levels'][1].to(args.device)
            level_2 = datas['levels'][2].to(args.device)
            c_1 = datas['c'][0].to(args.device)
            c_2 = datas['c'][1].to(args.device)
            c_3 = datas['c'][2].to(args.device)
            final_mat = datas['final_mat'].to(args.device)
            # load ground truth
            gt = gt.to(args.device)

            # training from the dataset
            pred = net(level_0, level_1, level_2, c_1, c_2, c_3, final_mat)

            # computing loss
            optimizer.zero_grad()
            loss_all = 0
            for i in range(pred.shape[0]):

                loss = segmentation_loss(pred[i], gt[i], datas['mesh_path'][i], weight=weight_train, device=args.device,
                                         loss_rate=args.loss_rate, bandwidth=args.bandwidth)

                if i != pred.shape[0] - 1:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()
                loss_all += loss.item()
                del loss
                torch.cuda.empty_cache()

            optimizer.step()

            # predicting pre-vertex labels
            pred_label = F.log_softmax(pred, dim=2)

            preds.append(pred_label.cpu())
            labels.append(gt.cpu())

            # computing the accuracy
            acc = 0
            for i in range(pred.shape[0]):
                acc += accuracy(pred_label[i][gt[i] != -1].cpu(), gt[i][gt[i] != -1].cpu()).item()

            loss_all /= pred.shape[0]
            acc /= pred.shape[0]

            train_loss_avg.update(loss_all)
            train_acc_avg.update(acc)

            t.set_postfix(loss='{:05.4f}'.format(train_loss_avg()))
            t.update()

            del level_0, level_1, level_2, c_1, c_2, final_mat, c_3
            torch.cuda.empty_cache()

    # tensorboard log: train
    writer.add_scalar('Accuracy/train_vertices', train_acc_avg(), epoch)
    writer.add_scalar('Loss/train_vertices', train_loss_avg(), epoch)

    return train_loss_avg(), train_acc_avg()


def test(args, net, test_dl, weight_test, writer=None, epoch=None):
    net.eval()

    preds = []
    labels = []

    test_loss_avg = RunningAverage()
    test_acc_avg = RunningAverage()
    test_vertex_acc_avg = RunningAverage()

    accuracy = Accuracy()

    with torch.no_grad():
        for j, (datas, gt, gt_face, vf) in enumerate(test_dl):
            # load test data
            level_0 = datas['levels'][0].to(args.device)
            level_1 = datas['levels'][1].to(args.device)
            level_2 = datas['levels'][2].to(args.device)
            c_1 = datas['c'][0].to(args.device)
            c_2 = datas['c'][1].to(args.device)
            c_3 = datas['c'][2].to(args.device)
            final_mat = datas['final_mat'].to(args.device)
            # load ground truth
            gt = gt.to(args.device)

            pred = net(level_0, level_1, level_2, c_1, c_2, c_3, final_mat)

            pred_label = F.log_softmax(pred, dim=2)

            preds.append(pred_label.cpu())
            labels.append(gt.cpu())

            acc = 0
            acc_vertex = 0
            loss_all = 0

            # transform the gt of vertices to faces
            for i in range(vf.shape[0]):
                vf_labels = np.zeros(
                    (gt_face[i][gt_face[i] != -1].shape[0], 10))  # max_dim=10 > the number of the classes
                for vertex, faces in enumerate(vf[i]):
                    for f in faces:
                        if f == -1:
                            break
                        vf_labels[f, np.argmax(pred_label[i, vertex].cpu()).item()] += 1
                pred_face_labels = torch.from_numpy(np.argmax(vf_labels, axis=1))

                # acc for per-face
                acc += accuracy(pred_face_labels, gt_face[i][gt_face[i] != -1]).item()
                # acc for per-vertex
                acc_vertex += accuracy(pred_label[i][gt[i] != -1].cpu(), gt[i][gt[i] != -1].cpu()).item()

                loss_all += segmentation_loss(pred[i], gt[i], datas['mesh_path'][i], weight=weight_test,
                                              device=args.device, loss_rate=args.loss_rate,
                                              bandwidth=args.bandwidth).item()

                # visualizing face labels
                if args.mode == 'test':
                    faces_label(datas['mesh_path'][i], pred_face_labels, args.name, args.num_classes)

            loss_all /= pred.shape[0]
            acc /= pred.shape[0]
            acc_vertex /= pred.shape[0]

            test_loss_avg.update(loss_all)
            test_acc_avg.update(acc)
            test_vertex_acc_avg.update(acc_vertex)

            del level_0, level_1, level_2, c_1, c_2, final_mat, c_3
            torch.cuda.empty_cache()

    if args.mode == 'train':
        # tensorboard log: test
        writer.add_scalar('Accuracy/val_faces', test_acc_avg(), epoch)
        writer.add_scalar('Accuracy/val_vertices', test_vertex_acc_avg(), epoch)
        writer.add_scalar('Loss/val_vertices', test_loss_avg(), epoch)

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
    parser.add_argument('--scheduler_mode', choices=['CosWarm', 'MultiStep', 'Warmup'], default='CosWarm')
    parser.add_argument('--scheduler_T0', type=int, default=30)
    parser.add_argument('--scheduler_eta_min', type=float, default=3e-7)
    parser.add_argument('--warm_up_T_max', type=int, default=60)
    parser.add_argument('--warm_up_epochs', type=int, default=20)
    parser.add_argument('--weight_decay', type=float, default=0.3)
    parser.add_argument('--amsgrad', action='store_true')
    parser.add_argument('--loss_rate', type=float, default=5e-3)
    parser.add_argument('--bandwidth', type=float, default=1.0)
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--num_inputs', nargs='+', default=[512, 128, 32], type=int, help='Multi-resolution input')
    parser.add_argument('--data_path', type=str, default='data/noise_data')
    parser.add_argument('--name', type=str, default='humanbody_input512_128_32_batchSize128_T50_lossRate5e3')

    args = parser.parse_args()
    print(args)

    # setting the seed
    same_seed(args.seed)

    print('Load Dataset...')
    base_dataset = BaseDataset(args)
    if args.mode == 'train':
        train_dl, test_dl, weight_train, weight_test = base_dataset.segDataset()
    else:
        test_dl, weight_test = base_dataset.segDataset()

    # define the Net
    net = Net(args.num_classes).to(args.device)

    # path to save the checkpoints
    if not os.path.exists(os.path.join('checkpoints', args.name)):
        os.makedirs(os.path.join('checkpoints', args.name))

    # path to save the visualization results
    if args.mode == 'test' and (not os.path.exists(os.path.join('visualization_result', args.name))):
        os.makedirs(os.path.join('visualization_result', args.name))

    min_val_loss = np.inf
    max_val_acc = -np.inf

    if args.mode == 'train':

        # Use wandb to visualize the training process
        wandb.init(project='lap_seg', entity='laplacian2mesh', config=args, name=args.name, sync_tensorboard=True)
        wandb.watch(net, log="gradients", log_graph=False)
        # tensorboard
        writer = SummaryWriter(os.path.join('checkpoints', args.name, 'log_dir'))

        # Network initialization
        init_weights(net)

        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)

        # select scheduler mode
        scheduler = None
        if args.scheduler_mode == 'CosWarm':
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.scheduler_T0, T_mult=2,
                                                                 eta_min=args.scheduler_eta_min, verbose=True)
        elif args.scheduler_mode == 'MultiStep':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 100], gamma=0.1, verbose=True)
        elif args.scheduler_mode == 'Warmup':
            warmup_cosine_lr = warm_up_with_cosine_lr(args.warm_up_epochs, args.scheduler_eta_min, args.lr,
                                                      args.epochs, args.warm_up_T_max)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lr)

        # training
        for epoch in tqdm.trange(args.epochs):
            train_loss, train_acc = train(args, net, train_dl, weight_train, writer, epoch)
            test_loss, test_acc = test(args, net, test_dl, weight_test, writer, epoch)

            writer.add_scalar('Utils/lr_scheduler', scheduler.get_last_lr()[0], global_step=epoch)
            scheduler.step()

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
    # test
    else:
        net.load_state_dict(torch.load(os.path.join('checkpoints', args.name, 'best_acc.pth')))
        test_loss, test_acc = test(args, net, test_dl, weight_test)
        save_logging(args, test_loss, test_acc)
