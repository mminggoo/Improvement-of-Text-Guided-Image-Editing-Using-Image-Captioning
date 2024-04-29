import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm
from einops import rearrange, repeat
from omegaconf import OmegaConf

from diffusers import DDIMScheduler

from masactrl.diffuser_utils import MasaCtrlPipeline, MasaCtrlPipeline5
from masactrl.masactrl_utils import AttentionBase
from masactrl.masactrl_utils import regiter_attention_editor_diffusers

from torchvision.utils import save_image
from torchvision.io import read_image
from pytorch_lightning import seed_everything

from transformers import AutoProcessor, BlipForConditionalGeneration, Blip2ForConditionalGeneration

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--scale", type=float)
args = parser.parse_args()

output_dir_root = f'../workdir_objecton/masa3_10_4_interpolate0.8{args.scale}/'

text_interpolate_scale = 0.8

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model_path = "xyn-ai/anything-v4.0"
model_path = "CompVis/stable-diffusion-v1-4"
# model_path = "runwayml/stable-diffusion-v1-5"
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
model = MasaCtrlPipeline3.from_pretrained(model_path, scheduler=scheduler).to(device)


model_id = "Salesforce/blip-image-captioning-base"
model_id = "BLIP_captioning"

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
captioner = BlipForConditionalGeneration.from_pretrained(model_id).to(device)

from masactrl.masactrl import MutualSelfAttentionControl, MutualSelfAttentionControlMaskAuto
from torchvision.io import read_image
from PIL import Image

def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image


seed = 42
seed_everything(seed)

from torchmetrics.functional.multimodal import clip_score
from functools import partial
import lpips
from torchvision import transforms
import json

file_path = '../eval/tedbench/input_list.json'

# JSON 파일 읽기
with open(file_path, 'r') as file:
    TEDbench_datas = json.load(file)


clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
clip_score_fn_large = partial(clip_score, model_name_or_path="openai/clip-vit-large-patch14")
loss_fn_alex = lpips.LPIPS(net='alex') # alex vgg
loss_fn_vgg = lpips.LPIPS(net='vgg') # alex vgg

def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int), prompts).detach()
    clip_score2 = clip_score_fn_large(torch.from_numpy(images_int), prompts).detach()
    return round(float(clip_score), 4), round(float(clip_score2), 4)
def calculate_lpips(img_paths0, img_paths1):
    convert_tensor = transforms.ToTensor()

    vgg_scores = []
    alex_scores = []
    for img_path1, img_path2 in zip(img_paths0, img_paths1):
        image1 = convert_tensor(Image.open(img_path1).convert("RGB").resize((256, 256))).unsqueeze(0)
        image2 = convert_tensor(Image.open(img_path2).convert("RGB").resize((256, 256))).unsqueeze(0)
        vgg_lpips_score = loss_fn_vgg(image1, image2).item() # shape (1,3,512,512)
        alex_lpips_score = loss_fn_alex(image1, image2).item() # shape (1,3,512,512)
        vgg_scores.append(vgg_lpips_score)
        alex_scores.append(alex_lpips_score)
    return np.mean(vgg_scores), np.mean(alex_scores)

root_path = '../eval/tedbench/originals/'

source_images = []
eval_images = []
lpips_images = []
eval_prompts = []

for data in TEDbench_datas:
    # source image
    SOURCE_IMAGE_PATH = root_path + data['img_name']
    source_image = load_image(SOURCE_IMAGE_PATH, device)
    source_images.append(SOURCE_IMAGE_PATH)

    source_prompt = ""

    # invert the source image
    start_code, latents_list = model.invert(source_image,
                                            source_prompt,
                                            guidance_scale=7.5,
                                            num_inference_steps=50,
                                            return_intermediates=True)
    start_code = start_code.expand(2, -1, -1, -1)

    out_dir = output_dir_root
    os.makedirs(out_dir, exist_ok=True)
    sample_count = len(os.listdir(out_dir))
    out_dir = os.path.join(out_dir, f"sample_{sample_count}")
    os.makedirs(out_dir, exist_ok=True)

    inputs = processor(Image.open(SOURCE_IMAGE_PATH), 'object on ', return_tensors="pt").to(device)
    out = captioner.generate(**inputs)

    target_prompt = data['target_text'] + ',' + processor.decode(out[0], skip_special_tokens=True)
    eval_prompts.append(data['target_text'])
    prompts = [source_prompt, target_prompt]

    # inference the synthesized image with MasaCtrl
    STEP = 4
    LAYPER = 10

    # hijack the attention module
    editor = MutualSelfAttentionControl(STEP, LAYPER)
    regiter_attention_editor_diffusers(model, editor)

    # inference the synthesized image
    image_masactrl = model(prompts,
                          latents=start_code,
                          guidance_scale=7.5,
                          text_interpolate_scale=text_interpolate_scale,
                          latent_interpolate_scale=args.scale,)

    # save the synthesized image
    out_image = torch.cat([source_image * 0.5 + 0.5,
                          image_masactrl[0:1],
                          image_masactrl[-1:]], dim=0)
    save_image(out_image, os.path.join(out_dir, f"all_{SOURCE_IMAGE_PATH.split('/')[-1]}_{target_prompt}.png"))
    save_image(out_image[0], os.path.join(out_dir, f"source_step{STEP}_layer{LAYPER}.png"))
    save_image(out_image[1], os.path.join(out_dir, f"reconstructed_source_step{STEP}_layer{LAYPER}.png"))
    save_image(out_image[2], os.path.join(out_dir, f"{SOURCE_IMAGE_PATH.split('/')[-1]}_{target_prompt}.png"))

    eval_images.append(out_image[2].cpu().detach().numpy())
    lpips_images.append(os.path.join(out_dir, f"{SOURCE_IMAGE_PATH.split('/')[-1]}_{target_prompt}.png"))

tedbench_clip_score, tedbench_clip_score2 = calculate_clip_score(np.array(eval_images), eval_prompts)
tedbench_vgg_lpips_score, tedbench_alex_lpips_score = calculate_lpips(lpips_images, source_images)
print(f'Clip Score: {tedbench_clip_score / 100}')
print(f'LPIPS Score vgg: {tedbench_vgg_lpips_score}, alex: {tedbench_alex_lpips_score}')


from torchmetrics.functional.multimodal import clip_score
from functools import partial
import lpips
from torchvision import transforms
import pandas as pd

file_path = '../eval/TEdBench_plusplus/tedbench++.csv'

TEDbenchplusplus_datas = pd.read_csv(file_path)


root_path = '../eval/TEdBench_plusplus/originals/'

source_images = []
eval_images = []
lpips_images = []
eval_prompts = []

for i in range(len(TEDbenchplusplus_datas)):
    data = TEDbenchplusplus_datas.iloc[i]

    # source image
    SOURCE_IMAGE_PATH = root_path + data['img_name']
    source_image = load_image(SOURCE_IMAGE_PATH, device)
    source_images.append(SOURCE_IMAGE_PATH)

    source_prompt = ""

    # invert the source image
    start_code, latents_list = model.invert(source_image,
                                            source_prompt,
                                            guidance_scale=7.5,
                                            num_inference_steps=50,
                                            return_intermediates=True)
    start_code = start_code.expand(2, -1, -1, -1)

    out_dir = output_dir_root
    os.makedirs(out_dir, exist_ok=True)
    sample_count = len(os.listdir(out_dir))
    out_dir = os.path.join(out_dir, f"sample_{sample_count}")
    os.makedirs(out_dir, exist_ok=True)

    inputs = processor(Image.open(SOURCE_IMAGE_PATH), 'object on ', return_tensors="pt").to(device)
    out = captioner.generate(**inputs)

    target_prompt = data['target_text'] + ',' + processor.decode(out[0], skip_special_tokens=True)
    eval_prompts.append(data['target_text'])
    prompts = [source_prompt, target_prompt]

    # inference the synthesized image with MasaCtrl
    STEP = 4
    LAYPER = 10

    # hijack the attention module
    editor = MutualSelfAttentionControl(STEP, LAYPER)
    regiter_attention_editor_diffusers(model, editor)

    # inference the synthesized image
    image_masactrl = model(prompts,
                          latents=start_code,
                          guidance_scale=7.5,
                          text_interpolate_scale=text_interpolate_scale,
                          latent_interpolate_scale=args.scale,)

    # save the synthesized image
    out_image = torch.cat([source_image * 0.5 + 0.5,
                          image_masactrl[0:1],
                          image_masactrl[-1:]], dim=0)
    save_image(out_image, os.path.join(out_dir, f"all_{SOURCE_IMAGE_PATH.split('/')[-1]}_{target_prompt}.png"))
    save_image(out_image[0], os.path.join(out_dir, f"source_step{STEP}_layer{LAYPER}.png"))
    save_image(out_image[1], os.path.join(out_dir, f"reconstructed_source_step{STEP}_layer{LAYPER}.png"))
    save_image(out_image[2], os.path.join(out_dir, f"{SOURCE_IMAGE_PATH.split('/')[-1]}_{target_prompt}.png"))

    eval_images.append(out_image[2].cpu().detach().numpy())
    lpips_images.append(os.path.join(out_dir, f"{SOURCE_IMAGE_PATH.split('/')[-1]}_{target_prompt}.png"))

tedbenchplus_clip_score, tedbenchplus_clip_score2 = calculate_clip_score(np.array(eval_images), eval_prompts)
tedbenchplus_vgg_lpips_score, tedbenchplus_alex_lpips_score = calculate_lpips(lpips_images, source_images)
print(f'Clip Score: {tedbenchplus_clip_score / 100}')
print(f'LPIPS Score vgg: {tedbenchplus_vgg_lpips_score}, alex: {tedbenchplus_alex_lpips_score}')

file_path = os.path.join(output_dir_root, "result.txt")

with open(file_path, "w") as file:
    file.write(f'Tedbench\n')
    file.write(f'Clip Score: {tedbench_clip_score / 100}\n')
    file.write(f'Clip Score2: {tedbench_clip_score2 / 100}\n')
    file.write(f'LPIPS Score vgg: {tedbench_vgg_lpips_score}\n')
    file.write(f'LPIPS Score alex: {tedbench_alex_lpips_score}\n')
    file.write(f'Tedbench++\n')
    file.write(f'Clip Score: {tedbenchplus_clip_score / 100}\n')
    file.write(f'Clip Score2: {tedbenchplus_clip_score2 / 100}\n')
    file.write(f'LPIPS Score vgg: {tedbenchplus_vgg_lpips_score}\n')
    file.write(f'LPIPS Score alex: {tedbenchplus_alex_lpips_score}\n')