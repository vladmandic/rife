import time
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from model_rife import RifeModel


model: RifeModel = None


def load(model_path = '../model/flownet-v46.pkl'):
    global model # pylint: disable=global-statement
    if model is None:
        model = RifeModel()
        model.load_model(model_path, -1)
        model.eval()
        model.device()


def interpolate(image0: np.ndarray, image1: np.ndarray, count: int = 1):
    if model is None:
        load()
    interpolated = []
    w, h, _c = image0.shape

    def execute(I0, I1, n):
        if model.version >= 3.9:
            res = []
            for i in range(n):
                res.append(model.inference(I0, I1, (i+1) * 1. / (n+1), scale=1.0))
            return res
        else:
            middle = model.inference(I0, I1, scale=1.0)
            if n == 1:
                return [middle]
            first_half = execute(I0, middle, n=n//2)
            second_half = execute(middle, I1, n=n//2)
            if n % 2:
                return [*first_half, middle, *second_half]
            else:
                return [*first_half, *second_half]

    def f_pad(img):
        return F.pad(img, padding).to(torch.float16) # pylint: disable=not-callable

    tmp = max(128, int(128 / 1.0))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    with torch.no_grad():
        I0 = f_pad(torch.from_numpy(np.transpose(image0, (2,0,1))).to('cuda', non_blocking=True).unsqueeze(0).float() / 255.)
        I1 = f_pad(torch.from_numpy(np.transpose(image1, (2,0,1))).to('cuda', non_blocking=True).unsqueeze(0).float() / 255.)
        output = execute(I0, I1, count)
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            interpolated.append(mid[:h, :w])

    return interpolated


if __name__ == '__main__':
    print('loading')
    load('../model/flownet-v46.pkl')
    image0 = cv2.imread('image0.jpg')
    image1 = cv2.imread('image1.jpg')
    print('images', image0.shape, image1.shape)
    for n in range(10):
        t0 = time.time()
        interpolated = interpolate(image0, image1, count=1)
        t1 = time.time()
        for img in interpolated:
            print('interpolated:', img.shape, round(t1 - t0, 4))
