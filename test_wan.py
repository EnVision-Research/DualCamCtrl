import torch
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import cv2
from PIL import Image
from typing import List, Optional
import numpy as np


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP",
            origin_file_pattern="diffusion_pytorch_model*.safetensors",
            offload_device="cpu",
        ),
        ModelConfig(
            model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP",
            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
            offload_device="cpu",
        ),
        ModelConfig(
            model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP",
            origin_file_pattern="Wan2.1_VAE.pth",
            offload_device="cpu",
        ),
        ModelConfig(
            model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP",
            origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            offload_device="cpu",
        ),
    ],
)
pipe.enable_vram_management()
T = [i for i in range(10, 40)]
for t in T:
    video = np.random.randint(0, 255, (t, 480, 640, 3), dtype=np.uint8)
    input_video = pipe.preprocess_video(video)
    pipe.vae.to("cuda")
    print(f"Shape of input video: {input_video.shape}")
    with torch.no_grad():
        output = pipe.vae.encode(input_video, device="cuda")

    output = output
    print(f"Shape of output: {output.shape}")
