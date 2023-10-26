import os
import json
import random
from PIL import Image

import torch 
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from datasets import load_dataset

from data.utils import pre_caption


class CocoCaptionDataset(Dataset):
    def __init__(self, split, tokenizer, processor, 
                 device=torch.device('cuda'), sample_size=None, hf_path='HuggingFaceM4/COCO'):
        self.raw_dataset = load_dataset(hf_path, split=split)
        if sample_size:
            random_indices = random.sample(range(len(self.raw_dataset)), sample_size)
            self.raw_dataset = self.raw_dataset.select(random_indices)
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        item = self.raw_dataset[idx]
        # Tokenize the text
        tokenized_text = self.tokenizer(item['sentences']["raw"], return_tensors="pt", truncation=True, padding='max_length', max_length=40)
        # Process the image
        processed_image = self.processor(item["image"], return_tensors="pt")
        return ({
            "input_ids": tokenized_text["input_ids"].squeeze(0).to(self.device),
            "attention_mask": tokenized_text["attention_mask"].squeeze(0).to(self.device),
            "pixel_values": processed_image["pixel_values"].squeeze(0).to(self.device),
        }, item['sentences']["raw"], item['imgid'])
