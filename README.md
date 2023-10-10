# RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation

Creates sequence of interpolated frames between given input images  

## Run

> python interpolate.py --input samples/ --output frames/ --buffer 25 --multi 25

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

Both examples are created using [SD.Next](https://github.com/vladmandic/automatic)

### Using [AnimateDiff](https://github.com/yownas/seed_travel) extension

https://github.com/vladmandic/rife/assets/57876960/65cf5c7d-e376-4ca9-b03e-2c81d2d79b2f

[Video](./samples/rife.mp4): *2.5sec at 25fps using 16 input images*

### Using [Seed Travel](https://github.com/yownas/seed_travel) extension

https://github.com/vladmandic/rife/assets/57876960/22ca5650-a770-4adb-b846-6dc06cdc3b26

[Video](./samples/video.mp4): *9sec at 30fps using 10 input images*
![Inputs](./samples/grid.jpg)

## Credits

- <https://github.com/megvii-research/ECCV2022-RIFE>
- <https://github.com/hzwer/Practical-RIFE>
