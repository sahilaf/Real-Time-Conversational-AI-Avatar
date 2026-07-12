# SyncTalk_2D

SyncTalk_2D is a 2D lip-sync video generation model based on SyncTalk and [Ultralight-Digital-Human](https://github.com/anliyuan/Ultralight-Digital-Human). It can generate lip-sync videos with high quality and low latency, and it can also be used for real-time lip-sync video generation.

Compared to the Ultralght-Digital-Human, we have improved the audio feature encoder and increased the resolution to 328 to accommodate higher-resolution input video. This version can realize high-definition, commercial-grade digital humans.

与Ultralght-Digital-Human相比，我们改进了音频特征编码器，并将分辨率提升至328以适应更高分辨率的输入视频。该版本可实现高清、商业级数字人。

## Setting up
Set up the environment
``` bash
conda create -n synctalk_2d python=3.10
conda activate synctalk_2d
```
``` bash
# install dependencies
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge ffmpeg  #very important
pip install opencv-python transformers soundfile librosa onnxruntime-gpu configargparse
pip install numpy==1.23.5
```

## Prepare your data
1. Record a 5-minute video with your head facing the camera and without significant movement. At the same time, ensure that the camera does not move and the background light remains unchanged during video recording.
2. Don't worry about FPS, the code will automatically convert the video to 25fps.
3. No second person's voice can appear in the recorded video, and a 5-second silent clip is left at the beginning and end of the video.
4. Don't wear clothes with overly obvious texture, it's better to wear single-color clothes.
5. The video should be recorded in a well-lit environment.
6. The audio should be clear and without background noise.


## Train
1. put your video in the 'dataset/name/name.mp4' 

- example: dataset/May/May.mp4

2. run the process and training script

``` bash
bash training_328.sh name gpu_id
```
- example: bash training_328.sh May 0

- Waiting for training to complete, approximately 5 hours

- If OOM occurs, try reducing the size of batch_size

## Inference

``` bash
python inference_328.py --name data_name --audio_path path_to_audio.wav
```
- example: python inference_328.py --name May --audio_path demo/talk_hb.wav

- the result will be saved in the 'result' folder

## Acknowledgements
This code is based on [Ultralight-Digital-Human](https://github.com/anliyuan/Ultralight-Digital-Human) and [SyncTalk](https://github.com/ZiqiaoPeng/SyncTalk). We thank the authors for their excellent work.
