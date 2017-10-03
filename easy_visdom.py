import os
import visdom

from functools import partial

import numpy as np


def _check_length(*inputs):
    lens = [len(i) for i in inputs]

    # All have the same length.
    assert lens[1:] == lens[:-1], \
        """statistics' names and statistics must have the same length.
        While received {} respectively""".format(lens)
    return lens[0]


class EasyVisdom:
    def __init__(self, from_scratch,
                 total_i,
                 start_i=1,
                 stats=['loss', 'acc', 'dice_overlap'],
                 mode=['train', 'val'],
                 results_dir='./results',
                 env='main'):
        self.start_i = start_i
        self.stats = stats
        self.names = np.array([x + '_' + y for x in mode for y in stats])
        self.results_dict = {name: np.zeros(total_i) for name in self.names}
        self.wins = []  # Windows for scalars
        self.wins_val_im = []  # Windows for images
        self.wins_train_im = []
        self.vis = visdom.Visdom(env=env)

        # Clear previous windows
        self.vis.close()

        if not from_scratch:
            # Load statistics from `.npy`
            pre_dict = np.load(os.path.join(results_dir, 'results_dict.npy')).item()
            pre_size = np.count_nonzero(pre_dict[self.names[0]])
            cur_size = total_i

            # In case the capacity of dict changed
            for key, value in self.results_dict.items():
                if pre_size < cur_size:  # increase, copy previous records
                    value[:pre_size] = pre_dict[key][:pre_size]
                else:  # decrease, the beginning records will be discarded
                    offset = pre_size - cur_size
                    value[:cur_size] = pre_dict[key][offset: pre_size]

    def vis_scalar(self, step, train_statistics, val_statistics):
        len_stat = _check_length(train_statistics, val_statistics, self.stats)

        for j, stat in enumerate(self.names[: len_stat]):
            self.results_dict[stat][step - 1] = train_statistics[j]

        for j, stat in enumerate(self.names[len_stat:]):
            self.results_dict[stat][step - 1] = val_statistics[j]

        epoch_results = np.array(list(zip(train_statistics, val_statistics)))
        basic_opts = partial(dict, xlabel='Epoch', legend=['train', 'val'])

        if step == self.start_i:
            # Resume values from records
            if step > 1:
                record_results = [np.column_stack(
                        (self.results_dict['train_' + stat][:step], self.results_dict['val_' + stat][:step]))
                    for stat in self.stats]
                for j, stat in enumerate(self.stats):
                    self.wins.append(self.vis.line(X=np.arange(step), Y=record_results[j], opts=basic_opts(title=stat)))

            # Plots from scratch
            elif step == 1:
                for j, stat in enumerate(self.stats):
                    self.wins.append(
                            self.vis.line(X=np.array([step]), Y=epoch_results[None, j], opts=basic_opts(title=stat)))

        else:
            for j, win in enumerate(self.wins):
                self.vis.updateTrace(X=np.array([step]), Y=epoch_results[j][0, None], win=win, name='train')
                self.vis.updateTrace(X=np.array([step]), Y=epoch_results[j][1, None], win=win, name='val')

    def vis_images(self, step, train_images=None, val_images=None, im_titles=['input', 'label', 'prediction'],
                   show_interval=5):
        """
        Arguments:
            + step: current step
            + train_images: corresponds to titles, i.e. [train_image, train_true, train_pred]
            + val_images: corresponds to titles, i.e. [val_image, val_true, val_pred]
            + im_titles: titles of windows
            + show_interval: refresh frequency
        """
        _check_length(train_images, val_images, im_titles);

        if step == self.start_i:
            # Windows for images
            self.wins_train_im = [self.vis.images(item, opts=dict(title='train_' + im_titles[j]))
                                  for j, item in enumerate(train_images)]
            self.wins_val_im = [self.vis.images(item, opts=dict(title='val_' + im_titles[j]))
                                for j, item in enumerate(val_images)]
        else:
            if step % show_interval == 0:
                for j, item in enumerate(train_images):
                    self.vis.images(item, opts=dict(title='train_' + im_titles[j], caption='Epoch' + str(step)),
                                    win=self.wins_train_im[j])
                for j, item in enumerate(val_images):
                    self.vis.images(item, opts=dict(title='val_' + im_titles[j], caption='Epoch' + str(step)),
                                    win=self.wins_val_im[j])
