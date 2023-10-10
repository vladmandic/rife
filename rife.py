#!/bin/env python

import _thread
import argparse
import os
import time
import shutil
import tempfile
import warnings
import subprocess
from queue import Queue
import filetype
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from rich import print # pylint: disable=redefined-builtin
from rich.pretty import install as install_pretty
from rich.traceback import install as install_traceback
from tqdm.rich import tqdm

from model.ssim import ssim_matlab
from model.RIFE_HDv3 import Model


model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
count = 0


def load(model_path: str = 'rife/flownet-v46.pkl', fp16: bool = False):
    global model # pylint: disable=global-statement
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if fp16:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Model()
    model.load_model(model_path, -1)
    model.eval()
    model.device()


def interpolate(args): # pylint: disable=redefined-outer-name
    print('start interpolate')
    t0 = time.time()
    if model is None:
        load(args.model, args.fp16)
    videogen = []
    if args.seq is None:
        for f in os.listdir(args.input):
            fn = os.path.join(args.input, f)
            if os.path.isfile(fn) and filetype.is_image(fn):
                videogen.append(fn)
    else:
        files = sorted(os.listdir(args.input))
        current = args.seq
        for f in files:
            seq = os.path.basename(f).split('-')[0]
            if seq.isdigit() and int(seq) == current:
                fn = os.path.join(args.input, f)
                videogen.append(fn)
                current += 1

    videogen = sorted(videogen)
    print(f'inputs: {len(videogen)} {[os.path.basename(f) for f in videogen]}')
    # videogen.sort(key=lambda x:int(os.path.basename(x[:-4])))
    frame = cv2.imread(videogen[0], cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
    h, w, _ = frame.shape
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    def write(output_dir, buffer):
        global count # pylint: disable=global-statement
        item = buffer.get()
        while item is not None:
            cv2.imwrite(f'{output_dir}/{count:0>6d}.jpg', item[:, :, ::-1])
            item = buffer.get()
            count += 1

    def execute(I0, I1, n):
        if model.version >= 3.9:
            res = []
            for i in range(n):
                res.append(model.inference(I0, I1, (i+1) * 1. / (n+1), args.scale))
            return res
        else:
            middle = model.inference(I0, I1, args.scale)
            if n == 1:
                return [middle]
            first_half = execute(I0, middle, n=n//2)
            second_half = execute(middle, I1, n=n//2)
            if n % 2:
                return [*first_half, middle, *second_half]
            else:
                return [*first_half, *second_half]

    def pad(img):
        return F.pad(img, padding).half() if args.fp16 else F.pad(img, padding) # pylint: disable=not-callable

    tmp = max(128, int(128 / args.scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    buffer = Queue(maxsize=8192)
    _thread.start_new_thread(write, (args.output, buffer))

    print(f'padded start: frames={args.buffer}')
    for _i in range(args.buffer): # fill starting frames
        buffer.put(frame)

    I1 = pad(torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
    with tqdm(total=len(videogen), desc='interpolate', unit='frame') as pbar:
        for f in videogen:
            frame = cv2.imread(f, cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
            I0 = I1
            I1 = pad(torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
            I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
            I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            if ssim > 0.99: # skip duplicate frames
                continue
            if ssim < args.change:
                output = []
                for _i in range(args.buffer): # fill frames if change rate is above threshold
                    output.append(I0)
                for _i in range(args.buffer):
                    output.append(I1)
            else:
                output = execute(I0, I1, args.multi-1)
            for mid in output:
                mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                buffer.put(mid[:h, :w])
            buffer.put(frame)
            pbar.update(1)

    print(f'padded end: frames={args.buffer}')
    for _i in range(args.buffer): # fill ending frames
        buffer.put(frame)
    while not buffer.empty():
        time.sleep(0.5)
    t1 = time.time()
    print(f'end interpolate: input={len(videogen)} frames={count} time={round(t1 - t0, 2)}')


def movie(args): # pylint: disable=redefined-outer-name
    # ffmpeg -hide_banner -loglevel info -hwaccel auto -y -framerate 30 -i "frames/%6d.jpg" -r 30 -vcodec libx264 -preset medium -crf 23 -vf minterpolate=mi_mode=blend,fifo -movflags +faststart video.mp4
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        print('ffmpeg not found')
        return
    ff_presets = {
        'x264': '-vcodec libx264 -preset medium -crf 23 -pix_fmt yuv420p',
        'x265': '-vcodec libx265 -preset faster -crf 28 -pix_fmt yuv420p',
        'vpx-vp9': '-vcodec libvpx-vp9 -crf 34 -b:v 0 -deadline realtime -cpu-used 4',
        'aom-av1': '-vcodec libaom-av1 -crf 28 -b:v 0 -usage realtime -cpu-used 8 -pix_fmt yuv444p',
        'prores_ks': '-vcodec prores_ks -profile:v 3 -vendor apl0 -bits_per_mb 8000 -pix_fmt yuv422p10le',
    }
    outfile = os.path.join(args.output, 'movie.mp4')
    ff_template = f"{ffmpeg} -hide_banner -loglevel error -hwaccel auto -y -framerate {args.fps} -start_number 0 -i \"{args.output}/%6d.jpg\" {ff_presets['x264']} -movflags +faststart -metadata author=\"sd.next\" \"{outfile}\""
    print('ffmpeg start: ', ff_template)
    result = subprocess.run(ff_template, text=True, capture_output=True, shell=True, env=os.environ, check=False)
    if len(result.stdout) > 0:
        print('ffmpeg stdout:', result.stdout)
    if len(result.stderr) > 0:
        print('ffmpeg stderr:', result.stderr)
    if args.rm:
        print(f'removing images: path={args.output}')
        for f in os.listdir(args.output):
            fn = os.path.join(args.output, f)
            if os.path.isfile(fn) and filetype.is_image(fn):
                os.remove(fn)
    print(f'ffmpeg end: file={outfile}')


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    install_pretty()
    install_traceback()
    print('starting rife')
    tmp_folder = os.path.join(tempfile.gettempdir(), f'rife-{time.strftime("%Y%m%d-%H%M%S")}')
    parser = argparse.ArgumentParser(description='interpolate video frames using RIFE')
    parser.add_argument('--model', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'model/flownet-v46.pkl')), help='path to model, default: %(default)s')
    parser.add_argument('--input', type=str, required=True, default=None, help='input directory containing images, default: %(default)s')
    parser.add_argument('--output', type=str, default=tmp_folder, help='output directory for interpolated images, default: %(default)s')
    parser.add_argument('--scale', type=float, default=1.0, help='scale factor for interpolated images, default: %(default)s')
    parser.add_argument('--multi', type=int, default=4, help='number of frames to interpolate between two input images, default: %(default)s')
    parser.add_argument('--buffer', type=int, default=2, help='number of frames to buffer on scene change, default: %(default)s')
    parser.add_argument('--change', type=float, default=0.3, help='scene change threshold (lower is more sensitive, default: %(default)s')
    parser.add_argument('--fp16', action='store_true', help='use float16 precision instead of float32, default: %(default)s')
    parser.add_argument('--mp4', action='store_true', help='create movie file from interpolated images, default: %(default)s')
    parser.add_argument('--fps', type=int, default=25, help='desired framerate, default: %(default)s')
    parser.add_argument('--seq', type=int, default=None, help='image sequence start number, default: %(default)s')
    parser.add_argument('--rm', action='store_true', help='remove interpolated images, default: %(default)s')
    args = parser.parse_args()
    print('args', args)
    assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
    interpolate(args)
    if args.mp4:
        movie(args)
