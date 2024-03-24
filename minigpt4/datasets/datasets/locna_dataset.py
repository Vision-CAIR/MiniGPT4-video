import os
import json
import pickle
import random
import time
import itertools

import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from torch.utils.data import Dataset
import webdataset as wds

from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset


class LocNaCOCODataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, min_len=60):
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.min_len = min_len
        self.data = self.create_data(ann_paths)

        self.instruction_pool = [
            '<Img><ImageHere></Img> Describe this image in detail.',
            '<Img><ImageHere></Img> Take a look at this image and describe what you notice.',
            '<Img><ImageHere></Img> Please provide a detailed description of the picture.',
            '<Img><ImageHere></Img> Could you describe the contents of this image for me?'
        ]

    def create_data(self, ann_paths):
        raw_data = []
        for ann_path in ann_paths:
            with open(ann_path, 'r') as f:
                raw_data.extend([json.loads(line) for line in f])

        data = []
        for d in raw_data:
            if len(d['caption'].split(' ')) < 60: continue
            data.append(
                {'caption': d['caption'],
                 'image_path': '{:012d}.jpg'.format(int(d['image_id']))
                }
            )
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        image = Image.open(os.path.join(self.vis_root, sample['image_path'])).convert("RGB")
        image = self.vis_processor(image)
        instruction = random.choice(self.instruction_pool)
        instruction = "###Human: {} ###Assistant: ".format(instruction)

        return {
            "image": image,
            "instruction_input": instruction,
            "answer": sample['caption'],
        }


