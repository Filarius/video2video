## Abandomed

This project is currently abandomed, no more fixes to get work with latest version of Automatic1111

# Automatic1111 Stable Diffusion WebUI Video2Video Extension

## Pluging for img2img video processing
- No more image files on hard disk.
- Video fps can be set as original, or changed.
- Now with latent space temporal blending.

Result saved to **output** folder **img2img-video** as MP4 file in H264 encoding (no audio). 

Added optional temporal blending for latent space. Applied per each step between previous and current frame.

Need a FFmpeg. For OS Windows implemented automatic installation of FFmpeg.

Under development, bugs applied.

## Dependencies

ffmpeg

skvideo (pip install sk-video)

## TODO

Bug: latent blending will work right only for batch_size = 1
