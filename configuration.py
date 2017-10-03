import os
import torch

from utils.others import swap_format

IM_SIZE = (1280, 1918)
IMAGES_PATH = sorted(os.listdir('./data/train/'))
LABELS_PATH = [swap_format(i) for i in IMAGES_PATH]


class Configuration:
    def __init__(self, prefix, cuda_device, from_scratch):
        self.cuda = torch.cuda.is_available()
        self.cuda_device = cuda_device

        self.batch_size = 4
        self.grad_acc_num = 8
        self.epochs = 70
        self.augment_size = 100
        # self.train_size = 4000
        # self.test_size = 1088
        self.learning_rate = 1e-4
        # self.lr_decay_epoch = 100
        self.loss_size = 1
        self.seed = 714
        self.threads = 4
        self.resume_step = -1
        self.from_scratch = from_scratch
        self.prefix = prefix
        self.use_bbox = False
        self.input_size = None

    def generate_dirs(self):
        self.result_dir = os.path.join('./results', self.prefix)
        self.checkpoint_dir = os.path.join('./checkpoints', self.prefix)
        # for d in [self.subdir, self.result_dir, self.checkpoint_dir]:
        #     if not os.path.exists(d):
        #         os.makedirs(d)
        [os.makedirs(d) for d in [self.result_dir, self.checkpoint_dir] if not os.path.exists(d)]
