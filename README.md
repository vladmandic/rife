# RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation

Creates sequence of interpolated frames between given input images  

## Run

> python interpolate.py --input samples/ --output frames/ --buffer 25 --multi 70

    interpolating 4 images
    image samples/image000.jpg ssim 0.99 buffer 25 frames
    image samples/image001.jpg ssim 0.54 create 69 frames
    image samples/image002.jpg ssim 0.45 create 69 frames
    image samples/image003.jpg ssim 0.55 create 69 frames
    image samples/image003.jpg ssim 0.99 buffer 25 frames
    frames 259 time 4.24

- Reads input images from `samples/` and writes output images to `frames/`  
- Number of generated frames will be 70x input frames
- Start and end will be buffered/padded with 25 frames

> ffmpeg -hide_banner -loglevel warning -hwaccel auto -y -framerate 30 -i "frames/%6d.jpg" -r 30 -vcodec libx264 -preset medium -crf 23 -vf minterpolate=mi_mode=blend,fifo -movflags +faststart samples/video.mp4

- Creates a video file from interpolated frames

## Options

> ./interpolate.py --help

    --model MODEL    path to model
    --input INPUT    input directory containing images
    --output OUTPUT  output directory for interpolated images
    --scale SCALE    scale factor for interpolated images
    --multi MULTI    number of frames to interpolate between two input images
    --buffer BUFFER  number of frames to buffer on scene change
    --change CHANGE  scene change threshold (lower is more sensitive
    --fp16           use float16 precision instead of float32

## Example

[Inputs](./samples/image.jpg): *4 images*
![Inputs](./samples/image.jpg)

[Video](./samples/video.mp4): *9sec at 30fps*

https://user-images.githubusercontent.com/57876960/216691061-97d3c0d1-9eb8-4753-8a5f-d0b8954bb216.mp4


*Note*: Images are generated using [Stable-Diffusion](https://github.com/vladmandic/automatic) with [seed-travel](https://github.com/yownas/seed_travel)

# Credits

- <https://github.com/megvii-research/ECCV2022-RIFE>
- <https://github.com/hzwer/Practical-RIFE>
