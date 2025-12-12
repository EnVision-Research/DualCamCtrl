## Training Process of DualCamCtrl 

### 1. Dataset 

#### 1.1 Download
We use the [RealEstate10K](https://google.github.io/realestate10k/) dataset to train our model.

#### 1.2 Captions Preprocessing 

We utilize the [Qwen-Image](https://github.com/QwenLM/Qwen3-VL) repo to preprocess the captions. Please refer to [this json](https://github.com/EnVision-Research/DualCamCtrl/blob/main/demo_dataset/test_captions/0a0a998c176713fd/captions.json) for a sample caption json. We generate the captions for every 60 images in the dataset to ensure that every sampled video has at least one caption. It's a bit slow, it tooks us for about 3-4 days to generate the captions for the whole dataset on 4 A6000 GPUs.

#### 1.3 Depth Generation

We utilize the [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) (VDA) to generate the depth for video (under the setting of FP16 and ViT-L to balance both speed and precision). You may refer to [this file](https://github.com/EnVision-Research/DualCamCtrl/blob/main/demo_dataset/test_video_depth_maps/0a0a998c176713fd/depth_vitl_fp16.mp4) for a sample video. It took us for about 2-3 days to generate the depth for the whole dataset on 4 A6000 GPUs.

#### 1.4 Format the dataset 

We organize the dataset as follows ( for split in {'test','train'} ):
```
Re10k 
├── {split}_meta.json            # meta data for each split
├── {split}_captions/            # captions for each split
├── {split}__video_depth_maps/   # depth for each split                
└── {split}_scenes/              # scenes for each split
```
You may run the dataset script to have a glance of the [demo dataset](../demo_dataset):
```
export PYTHONPATH=.
python examples/dataset/realestate10k.py
```

You are expected to get the following output:
```
images: XXX
images min: 0.0, max: 1.0
control: XXX
control min: 0.0, max: 0.9843137264251709
camera_infos: XXX
camera_infos min: -0.7278759479522705, max: 0.9999988675117493
prompt: The video depicts a cozy and well-decorated bedroom...
```




### 2. Training 


Some important args:
- training_state_dir: the directory which you want to load the ckpt from
- output_path: the directory where you save the ckpt
- freeze_main_except, freeze_control_except: the part of the model you want to freeze
- learning_rate: 3e-6, higher learning rate would lead to worse stability of training
- init_validate: whether to start a validate before the training
- drop_loss_rate: the rate of drop loss for the geometry signal
- use_image_depth: false, we don't use the image depth in the training process since it would lead to severe temporal discontinuity
- copy_control_weights: true at the decoupled stage and false at the fusion stage
- freeze_zero_linear: true at the decoupled stage and false at the fusion stage

#### 2.1 Deepspeed 

Since the model is large, we utilize deepspeed along with accelerate to train the whole model with ZeRO stage 2, refer [this yaml](../train_config/accelerate_config/accelerate_debug.yaml) for reference of setting.


#### 2.2 Two stage training
As we mentioned in the paper, the training process is divided into two stages. 

##### Decoupled Stage
```
bash train_script/I2V/train_depth.sh
```


##### Fusion Stage 

Setting the 'training_state_dir' args in [this yaml](../train_config/normal_config/i2v_train_depth.yaml) to the directory where you save the ckpt of Decoupled stage, then:
```
bash train_script/I2V/train_fuse.sh
```

#### 2.4 Text to Video 

This repo mainly focus on the image to video task, but we also provide the text to video task. You alter the 't2v' args in both yaml files to 'true' to enable the text to video task, save then as 't2v_train_depth.yaml' and 't2v_train_fuse_5_10_70_3e6.yaml' respectively, then run the bash script:

```
bash train_script/T2V/train_depth.sh
bash train_script/T2V/train_fuse.sh
```

#### 2.5 Training Details

We trained the model on 4 H100 GPUS with a effective batch size of 8 for both stage. For the first stages, it converged with 10k iterations, for the second stage, it will converged with ~70k iterations.


#### 2.6 Help us improve the training process description

Feel free to [open an issue](https://github.com/EnVision-Research/DualCamCtrl/issues) if you have any questions regarding the training process. It will help us for better implement the training process description to further benefit the community.


Best Regards,

All authors of DualCamCtrl