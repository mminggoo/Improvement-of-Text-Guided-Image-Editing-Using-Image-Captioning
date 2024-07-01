import torch
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CFG = {
    'IMG_SIZE':224,
    'epochs':2, # Epochs,
    'LR':5e-5, # Learning Rate,
    'batch_size':4, # Batch Size,
    'gradient_accumulation_steps': 4,
    'seed':42
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['seed']) # Seed 고정

from transformers import AutoProcessor, BlipForConditionalGeneration, Blip2ForConditionalGeneration
from transformers import TrainingArguments, Trainer

model_id = "Salesforce/blip-image-captioning-base"

processor = AutoProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
model.init_weights()

import datasets
from datasets import Dataset

train_data = pd.read_csv('./data.csv')

data_dict = {
    'image': train_data['image_file'].tolist(),
    'text': train_data['text'].tolist(),
}

train_dataset = Dataset.from_dict(data_dict)
train_dataset = train_dataset.cast_column('image', datasets.Image())

def transforms(example_batch):
    images = [x for x in example_batch["image"]]
    captions = [x for x in example_batch["text"]]
    inputs = processor(images=images, text=captions, padding="max_length")
    inputs.update({"labels": inputs["input_ids"]})
    return inputs

train_dataset.set_transform(transforms)
valid_dataset = train_dataset.select(list(range(1000)))


training_args = TrainingArguments(
    output_dir = "./logs",
    logging_dir = './logs',
    logging_steps = 50,
    learning_rate = CFG['LR'],
    num_train_epochs=CFG['epochs'],
    per_device_train_batch_size=CFG['batch_size'],
    per_device_eval_batch_size=CFG['batch_size'],
    gradient_accumulation_steps=CFG['gradient_accumulation_steps'],
    save_total_limit=2,
    remove_unused_columns=False,
    label_names=["labels"],
    load_best_model_at_end=True,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    seed=CFG['seed'],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

trainer.train()
