import os
import shutil

import numpy as np
import torch


def get_resume_path(checkpoint_dir, resume_step=-1, prefix='Untitled_'):
    """Return latest checkpoints by default otherwise return the specified one.
    
    Notice:
        Filenames in `checkpoint_dir` should be 'XXX_100.path', 'XXX_300.path'...

    Arguments:
    + checkpoint_dir:
    + prefix: str('XXX_'), name of checkpoint
    + resume_step: uint, indicates specific step to resume training.
    + path of the resumed checkpoint.

    Returns:
        resume path: path to resume checkpoint
        resume step: indicate step, used for plot
    """
    names = [os.path.join(checkpoint_dir, p) for p in os.listdir(checkpoint_dir) if 'best' not in p]
    require = os.path.join(checkpoint_dir, prefix + '_' + str(resume_step) + '.pth')
    best = require.replace(str(resume_step), 'best')

    if os.path.isfile(require):
        end_i = resume_step
        resume_path = require
    elif os.path.isfile(best) or resume_step == -1:
        latest = sorted(names, key=os.path.getmtime)[-1]
        end_i = int(latest.split('_')[-1][:-4])
        resume_path = best
    else:
        raise Exception('\'%s\' dose not exist!' % require)

    return resume_path, end_i


def load_checkpoints(model, checkpoint_dir, resume_step=-1, prefix='Untitled_'):
    """Load previous checkpoints"""

    cp, end_i = get_resume_path(checkpoint_dir, resume_step, prefix)
    pretrained_dict = torch.load(cp)
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    cp_name = os.path.basename(cp)
    print('---> Loading checkpoint {}...'.format(cp_name))
    return end_i + 1  # start_i


def save_checkpoints(model, checkpoint_dir, step, prefix='Untitled_', is_best=False):
    """Save 20 checkpoints at most"""

    names = [os.path.join(checkpoint_dir, n)  # keep best
             for n in os.listdir(checkpoint_dir) if 'best' not in n]
    # sort by time
    names = sorted(names, key=os.path.getmtime)
    if len(names) >= 5:
        os.remove(names[0])

    pattern = prefix + '_' + '{}' + '.pth'
    savepath = os.path.join(checkpoint_dir, pattern)
    # Recommend: save and load only the model parameters
    torch.save(model.state_dict(), savepath.format(step))
    print("===> ===> ===> Save checkpoint {} to {}".format(step, pattern.format(step)))
    if is_best:
        shutil.copyfile(savepath.format(step), savepath.format('best'))


def best_checkpoints(results_dir, checkpoint_dir, keys=['val_dice_overlap']):
    """Return the path of the best checkpoint. Typically is 'Not_Important_best.pth'
    If not exist, compute from rest checkpoints according to given keys.
    
    Assume:
        checkpoint's format is 'Not_Important_123.pth'
    """

    checkpoints = [os.path.join(checkpoint_dir, p) for p in os.listdir(checkpoint_dir) if 'best' not in p]

    # pattern == 'Not_Important_{}.pth' == 'Not_Important' + '_{}.pth'
    pattern = '_'.join(checkpoints[0].split('_')[: -1]) + '_{}.pth'
    best = pattern.format('best')

    if os.path.isfile(best):
        return best
    else:
        # '123' == '123.pth'[:-4] == ['Not', 'Important', '123.pth'][-1][:-4] == cp.split('_')[-1][:-4]
        resume_steps = [int(cp.split('_')[-1][:-4]) for cp in checkpoints]

        results_dict = np.load(os.path.join(results_dir, 'results_dict.npy')).item()
        selected_results = np.array([results_dict[key] for key in keys])
        sum_results = {step: selected_results[:, step - 1].sum() for step in resume_steps
                       if step > 0.7 * max(resume_steps)}

        best_checkpoints_step = sorted(sum_results, key=sum_results.get)[-1]
        return pattern.format(best_checkpoints_step)
