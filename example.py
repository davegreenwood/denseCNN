# %%

from torchvision.transforms import ToTensor, ToPILImage, Normalize
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from densecnn import Vgg16

NORM = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
TOTEN = ToTensor()
TOPIL = ToPILImage()


def crop_image(fname):
    img = Image.open(fname)
    w, h = img.size
    w, h = w // 32, h // 32
    w, h = w * 32, h * 32
    return img.crop([0, 0, w, h])


def nrm_img(fname):
    """Normalise an img and return tensor."""
    img = crop_image(fname)
    img = TOTEN(img)
    img = NORM(img)
    return img


def markers1(img):
    w, h = img.size
    return np.meshgrid([x for x in range(16, w, 32)],
                       [y for y in range(16, h, 32)])


# %%

fname = "building.jpg"
img = crop_image(fname)
ten = nrm_img(fname)
VGG = Vgg16()
out = VGG(ten[None, ...])

# %%

x, y = markers1(img)
rows, cols, features = out
xp, yp = cols.numpy(), rows.numpy()

fig, ax = plt.subplots(1, 1, figsize=[8, 5])
ax.imshow(img.convert("L").convert("RGB"))
ax.plot(x, y, lw=0, marker="+", color="g")

for a, b, c, d in zip(x, y, xp, yp):
    for i, j, dx, dy in zip(a, b, c-a, d-b):
        if dx == 0 and dy == 0:
            continue
        ax.arrow(i, j, dx, dy, length_includes_head=True,
                 head_width=3, head_length=3, color="y")

ax.plot(xp, yp, lw=0, marker="+", color="r")
plt.tight_layout()
fig.savefig("result.jpg", dpi=100)
