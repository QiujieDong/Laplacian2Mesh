import sys

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from dataset.dataset_cls import ClsDataset
from dataset.dataset_seg import segDataset


class BaseDataset(Dataset):
    def __init__(self, args):
        self.args = args
        super(BaseDataset, self).__init__()

    def classification_dataset(self):
        if self.args.mode == 'train':
            train_ds = ClsDataset(self.args, set_type='train', is_train=True)
            train_dl = DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True, pin_memory=False,
                                  num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)

        test_ds = ClsDataset(self.args, set_type='test', is_train=False)
        test_dl = DataLoader(test_ds, batch_size=self.args.batch_size, shuffle=False, pin_memory=False,
                             num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        if self.args.mode == 'train':
            return train_dl, test_dl
        else:
            return test_dl

    def segDataset(self):

        if self.args.mode == 'train':
            train_ds = segDataset(self.args, set_type='train', is_train=True)
            train_dl = DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True, pin_memory=False,
                                  num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)

            # Class_num is the ratio for solving the problem of data imbalance, which is used in the loss function.
            class_num_train = train_ds.class_num
            weight_train = class_num_train.max() / class_num_train

        test_ds = segDataset(self.args, set_type='test', is_train=False)
        test_dl = DataLoader(test_ds, batch_size=self.args.batch_size, shuffle=False, pin_memory=False,
                             num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor)
        # Class_num
        class_num_test = test_ds.class_num
        weight_test = class_num_test.max() / class_num_test

        if self.args.mode == 'train':
            return train_dl, test_dl, weight_train, weight_test
        else:
            return test_dl, weight_test
