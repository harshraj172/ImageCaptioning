import os
import json
import random
from PIL import Image

import torch 
from torch.utils.data import Dataset
# from torchvision.datasets.utils import download_url
from datasets import load_dataset

from data.utils import pre_caption


class CocoCaptionDataset(Dataset):
    def __init__(self, split, tokenizer, processor, eval_processor, max_length=64,
                 device=torch.device('cuda'), sample_size=None, hf_path='HuggingFaceM4/COCO', cache_dir='/p/scratch/ccstdl/raj3/imgcaption-data'):
        self.raw_dataset = load_dataset(hf_path, cache_dir=cache_dir, split=split)

        if sample_size:
            random_indices = random.sample(range(len(self.raw_dataset)), sample_size)
            self.raw_dataset = self.raw_dataset.select(random_indices)
        self.tokenizer = tokenizer
        self.processor = processor
        self.eval_processor = eval_processor
        self.device = device
        self.max_length = max_length
        
    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        item = self.raw_dataset[idx]
        # Tokenize the text
        tokenized_text = self.tokenizer(item['sentences']["raw"], return_tensors="pt", truncation=True, padding='max_length', max_length=self.max_length)
        # Process the image
        processed_image = self.processor(item["image"], return_tensors="pt")
        eval_processed_image = self.eval_processor(item['image'])
        return ({
            "input_ids": tokenized_text["input_ids"].squeeze(0),
            "attention_mask": tokenized_text["attention_mask"].squeeze(0),
            "pixel_values": processed_image["pixel_values"].squeeze(0),
            "eval_pixel_values": eval_processed_image.squeeze(0),
        }, item['sentences']["raw"], item['imgid'])
