import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from torch.autograd import Variable

from scipy.ndimage.interpolation import zoom


class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, relu=nn.ReLU()):
        """
        + Instantiate modules: conv-relu
        + Assign them as member variables
        """
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.relu = relu

    def forward(self, x):
        return self.relu(self.conv(x))


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, relu=nn.ReLU()):
        """
        + Instantiate modules: conv-bn-relu
        + Assign them as member variables
        """
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = relu

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class UpConcat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConcat, self).__init__()
        # Right hand side needs `Upsample`
        self.rhs_up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_fit = ConvBNReLU(in_channels + out_channels, out_channels)
        self.conv = nn.Sequential(ConvBNReLU(out_channels, out_channels), ConvBNReLU(out_channels, out_channels))

    def forward(self, lhs, rhs):
        rhs = self.rhs_up(rhs)
        rhs = make_same(lhs, rhs)
        cat = torch.cat((lhs, rhs), dim=1)
        return self.conv(self.conv_fit(cat))


# Block with shortcut
class DownBlock(nn.Module):
    """Reference: https://arxiv.org/pdf/1709.00201.pdf"""

    def __init__(self, in_channels, out_channels, ceil_mode=True):
        super(DownBlock, self).__init__()
        self.conv = nn.Sequential(ConvBNReLU(in_channels, in_channels), ConvBNReLU(in_channels, in_channels))
        self.pool = nn.Sequential(ConvBNReLU(in_channels, out_channels), nn.MaxPool2d(2, stride=2, ceil_mode=ceil_mode))

    def forward(self, x):
        conv = self.conv(x) + x
        return conv, self.pool(conv)


# Block with shortcut
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        # Right hand side needs `Upsample`
        self.conv = nn.Sequential(ConvBNReLU(in_channels + out_channels, in_channels),
                                  ConvBNReLU(in_channels, in_channels))
        self.up = nn.Sequential(ConvBNReLU(in_channels, out_channels), nn.UpsamplingBilinear2d(scale_factor=2))

    def forward(self, lhs, rhs):
        rhs = make_same(lhs, rhs)
        cat = torch.cat((lhs, rhs), dim=1)
        conv = self.conv(cat) + rhs
        return conv, self.up(conv)


class AtrousBlock(nn.Module):
    """Atrous Spatial Pyramid Pooling.

    Structure:
        1 x 1 conv  --> --> --> ↓
        3 x 3 conv (rate=1) --> --> ↓
        3 x 3 conv (rate=3) --> --> --> concat --> 1 x 1 conv --> PReLU
        3 x 3 conv (rate=6) --> --> ↑
        3 x 3 conv (rate=9) --> ↑
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, cat=True):
        super(AtrousBlock, self).__init__()
        self.conv_1r1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv_3r1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, dilation=1)
        self.conv_3r3 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=3, dilation=3)
        self.conv_3r6 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=6, dilation=6)
        self.conv_3r9 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=9, dilation=9)

        self.cat = cat  # if cat: in_channels = 1*4 + 1 = 5, else sum: in_channels = 0*4 + 1 = 1
        self.conv = ConvBNReLU((cat * 4 + 1) * out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        bulk = [self.conv_1r1(x), self.conv_3r1(x), self.conv_3r3(x), self.conv_3r6(x), self.conv_3r9(x)]
        if self.cat:
            out = torch.cat(bulk, dim=1)
        else:
            out = sum(bulk)
        return self.conv(out)


# Atrous down with shortcut
class DownAtrous(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownAtrous, self).__init__()
        self.conv = AtrousBlock(in_channels, in_channels)
        self.pool = nn.Sequential(ConvBNReLU(in_channels, out_channels), nn.MaxPool2d(2, stride=2, ceil_mode=True))

    def forward(self, x):
        conv = self.conv(x) + x
        return conv, self.pool(conv)


class ResNeXt(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, widen_factor=4):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXt, self).__init__()
        c = cardinality * out_channels // widen_factor
        self.conv_reduce = nn.Conv2d(in_channels, c, kernel_size=1, stride=1, padding=0)
        # self.bn_reduce = nn.BatchNorm2d(c)
        self.conv_conv = nn.Conv2d(c, c, kernel_size=3, stride=stride, padding=1, groups=cardinality)
        # self.bn = nn.BatchNorm2d(c)
        self.conv_expand = nn.Conv2d(c, out_channels, kernel_size=1, stride=1, padding=0)
        # self.bn_expand = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride > 1:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0))

    def forward(self, x):
        residual = self.shortcut(x)
        bottleneck = F.relu(self.conv_reduce(x))
        bottleneck = F.relu(self.conv_conv(bottleneck))
        bottleneck = self.conv_expand(bottleneck)
        return F.relu(residual + bottleneck)


class DualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DualBlock, self).__init__()
        self.residual = nn.Conv2d(in_channels, out_channels, 1)
        c = in_channels // 4
        self.conv = nn.Sequential(nn.Conv2d(in_channels, c, 1),
                                  ResNeXt(c, c, widen_factor=8),
                                  nn.Conv2d(c, out_channels, 1))
        self.fit = nn.Conv2d(in_channels + out_channels, out_channels, 1)

    def forward(self, x):
        conv = self.conv(x)
        concat = torch.cat([x, conv], dim=1)
        return F.relu(self.fit(concat) + self.residual(x))


class NLLLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(NLLLoss, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, outputs, targets):
        """
        Arguments:
            outputs: Variable torch.cuda.FloatTensor, maybe iterable (multi-loss)
            targets: Variable torch.cuda.FloatTensor, maybe iterable (multi-loss)
        """
        if not isinstance(outputs, (tuple, list)):
            return self.nll_loss(F.log_softmax(outputs[0]), targets[0].long().squeeze(1))

        loss = []  # multi-loss
        for i, output in enumerate(outputs):
            target = targets[i].long().squeeze(1)  # convert back to cuda variable
            loss.append(self.nll_loss(F.log_softmax(output), target))
        return sum(loss)


class DiceLoss(nn.Module):
    def __init__(self, weight=1):
        super(DiceLoss, self).__init__()
        self.w = weight * weight

    def forward(self, probs, trues):
        loss = []
        smooth = 1.  # (dim = )0 for Tensor result
        for i, prob in enumerate(probs):
            true = trues[i]
            intersection = torch.sum(self.w * prob * true, 0) + smooth
            union = torch.sum(self.w * prob * prob, 0) + torch.sum(self.w * true * true, 0) + smooth
            dice = 2.0 * intersection / union
            loss.append(1 - dice)
        return sum(loss)
        # return 1 - torch.clamp(dice, 0.0, 1.0 - 1e-7)


def make_same(good, evil):
    """
    good / evil could be 1-d, 2-d or 3-d Tensor, i.e., [batch_size, channels, (depth,) (height,) width]
    Implemented by tensor.narrow
    """
    # Make evil bigger
    g, e = good.size(), evil.size()
    ndim = len(e) - 2
    pad = int(max(np.subtract(g, e)))
    if pad > 0:
        pad = tuple([pad] * ndim * 2)
        evil = F.pad(evil, pad, mode='replicate')

    # evil > good:
    e = evil.size()  # update
    for i in range(2, len(e)):
        diff = (e[i] - g[i]) // 2
        evil = evil.narrow(i, diff, g[i])
    return evil


def weights_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal(m.weight)
            # init.constant(m.bias, 0.01)


def get_class_weight(data_loader):
    """To balance between foregrounds and background for NLL.

    Return:
        A Tensor consists [background_weight, *foreground_weight]
    """
    # Original label
    label = next(iter(data_loader))[-1].numpy()[:, 0]
    # Get elements in marks i.e., {0, 1}, {0, 10, 150, 250}...
    marks = np.unique(label)
    # The weight of each class
    weights = [(label == m).mean() for m in marks]
    # Inverse to rescale weights
    return 1 / torch.FloatTensor(weights)


def avg_class_weight(data_loader, avg_size=4):
    """Get average class weights.

    Return:
        A Tensor consists [background_weight, *foreground_weight]
    """
    weights = get_class_weight(data_loader)
    for i in range(avg_size - 1):
        weights += get_class_weight(data_loader)

    return weights.div(avg_size)


def multi_size(label, size=1):
    """Generate multi-size labels for multi-loss computation
    Arguments:
        label: [batch_size, 1, height, width] (torch.ByteTensor)
        size:  int or list indicates size
               E.g., size = 4 means reducing to x1, x2, x4, x8 size respectively
                     size = N means x1, x2, ..., x2^(N-1)
                     size = [1, 4, 16] means reducing to x1, x4, x16 size directly
    Returns:
        list of reduced-size labels (list of Variable torch.cuda.FloatTensor)
        [[batch_size, 1, height, width], [batch_size, 1, height/2, width/2], ...]
    """
    if isinstance(size, int):
        # size = [pow(2, x) for x in range(size)]
        diff = [2] * (size - 1)
    else:  # e.g., size = [1, 4, 16, 32], diff = [4, 4, 2]
        diff = np.divide(size[1:], size[:-1])

    labels = [Variable(label).cuda().float()]  # init
    label = label.numpy()
    for d in diff:
        factor = np.repeat([1, 1.0 / d], 2)
        label = zoom(label, factor, order=1, prefilter=False)
        labels.append(Variable(torch.from_numpy(label)).cuda().float())
    return labels


def active_flatten(outputs, targets, activation=F.softmax):
    """ Flatten, 2D --> 1D
    Arguments:
        outputs: [batch_size, 2, height, width] (torch.cuda.FloatTensor) Variable
        targets: [batch_size, 1, height, width] (torch.cuda.FloatTensor) Variable
        activation: according to loss function

    Return:
        preds: FloatTensor {0.0, 1.0} with shape [batch_size x (1) x height x width] Variable
        trues: FloatTensor {0.0, 1.0} with shape [batch_size x (1) x height x width] Variable
        probs: FloatTensor (0.0, 1.0) with shape [batch_size x (1) x height x width] Variable
    """
    if not isinstance(outputs, (tuple, list)):
        outputs.data = outputs.data.unsqueeze(0)

    preds, trues, probs = [], [], []
    for i, output in enumerate(outputs):
        output = output.permute(0, 2, 3, 1).contiguous()
        prob = activation(output.view(-1, 2))

        preds.append(prob.max(1)[1].float())
        trues.append(targets[i].view(-1))
        probs.append(prob[:, 1])
    return preds, trues, probs


def get_statistic(pred, true):
    """Compute dice among **positive** labels to avoid unbalance.

    Arguments:
        pred: FloatTensor {0.0, 1.0} with shape [batch_size, (1), height, width]
        true: FloatTensor {0.0, 1.0} with shape [batch_size, (1), height, width]

    Returns:
        tuple contains:
        + accuracy: (pred ∩ true) / true
        + mIOU:  pred ∩ true / (pred ∪ true) * 100
    """

    # Dice overlap
    pred = pred.eq(1).byte().data  # FloatTensor 0.0 / 1.0
    true = true.byte().data  # FloatTensor 0.0 / 1.0
    mIOU = (pred & true).sum() / (pred | true).sum() * 100

    # Accuracy
    acc = pred.eq(true).float().mean() * 100
    return acc, mIOU


def pre_visdom(image, label, pred, show_size=256):
    """Prepare (optional zoom) for visualization in Visdom.

    Arguments:
        image: torch.cuda.FloatTensor of size [batch_size, 3, height, width] Variable
        pred : torch.cuda.FloatTensor of size [batch_size * height * width ] Variable
        label: torch.ByteTensor of size [batch_size, 1, height, width]
        show_size: show images with size [batch_size, 3, height, width] in visdom

    Returns:
        image: numpy.array of size [batch_size, 3, height, width]
        label: numpy.array of size [batch_size, 1, height, width]
        pred : numpy.array of size [batch_size, 1, height, width]
    """
    pred = pred.view_as(label).mul(255)  # make label 1 to 255 for better visualization
    image, pred = [item.cpu().data.numpy() for item in [image, pred]]
    label = label.numpy()

    zoom_factor = np.append([1, 1], np.divide(show_size, image.shape[-2:]))
    return [zoom(item, zoom_factor, order=1, prefilter=False) for item in [image, label, pred]]
