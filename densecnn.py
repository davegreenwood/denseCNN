"""
Relocalise uniform features from VGG16.
Ref: https://arxiv.org/abs/1805.03879
"""

import torch
import torchvision.models as models


def max_index(x, rows, cols):
    """return the indices of the max pixel in each adjacent 2*2 region of x """
    idx_max = torch.argmax(
        torch.stack([x[0::2, 0::2], x[0::2, 1::2],
                     x[1::2, 0::2], x[1::2, 1::2]],
                    dim=-1), dim=-1)[rows, cols]
    max_rows, max_cols = idx_max // 2, idx_max % 2
    return 2 * rows + max_rows, 2 * cols + max_cols


def relocate(layers):
    """
    Layers is a list of the l2 norm of each maxpool result map,
    highest resolution first, lowest resolution last.
    For each feature in the lowest res map,
    returns the row, col coordinates of the max pixel in the highest res map.
    """
    n, m = layers[-1].shape[:2]
    rows, cols = torch.meshgrid(torch.arange(n), torch.arange(m))
    for x in reversed(layers[:-1]):
        rows, cols = max_index(x, rows, cols)
    return rows, cols


class Base(torch.nn.Module):
    """
    Relocate the max pixel from the lowest resolution max pool layer to the
    highest resolution layer.
    """

    def __init__(self):
        super(Base, self).__init__()
        self.idx = []
        self.features = []

    def forward(self, x):
        norm_layers, features = [], None
        for layer, model in enumerate(self.features):
            x = model(x)
            if layer in self.idx:
                norm_layers.append(torch.norm(x[0, ...], dim=0))
                features = x[0, ...]
        rows, cols = relocate(norm_layers)
        return rows, cols, features


class Vgg16(Base):
    """Relocate the max pixel from the lowest resolution max pool layer to the
    highest resolution layer."""

    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(models.vgg16(pretrained=True).features)
        self.idx = [2, 4, 9, 16, 23, 30]
        self.features = torch.nn.ModuleList(features).eval()
