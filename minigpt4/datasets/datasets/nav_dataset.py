import os
import json
import pickle
import math
import random
import glob
import torch
import time
import itertools

from torch.utils.data import Dataset
from PIL import Image, ImageDraw


class NavR2RDataset(Dataset):
    def __init__(self, vis_processor, text_processor, data_root):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.data_root = data_root
        self.data_ids = [subfolder.split('/')[-1] for subfolder in glob.glob(os.path.join(self.data_root, '*'))]

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.connect_sym = "!@#"

    def __len__(self):
        return len(self.data_ids)

    def preprocess(self, index):
        data_id = self.data_ids[index]
        with open(os.path.join(self.data_root, data_id, 'data.json'), 'r') as f:
            meta_data = json.load(f)

        instructions = meta_data['instructions']
        actions = meta_data['action']

        frames = []
        for i in range(meta_data['n_steps']):
            image_path = os.path.join(self.data_root, data_id, '{}.jpg'.format(i))
            frames.append(self.vis_processor(Image.open(image_path).convert("RGB")))

        return {
            "frames": frames,
            "instructions": instructions,
            "actions": actions,
            "data_id": data_id,
        }

    def __getitem__(self, index):
        data = self.preprocess(index)
        instruction = random.choice(data['instructions'])
        instruction = "Command: {}\n\n".format(instruction)

        obs = self.connect_sym.join(['<ImageHere> A: ' for _ in data['actions']])
        obs = instruction + obs
        act = self.connect_sym.join(data['actions'])

        stacked_frames = torch.stack(data["frames"][:-1], dim=0)

        return {
            "image": stacked_frames,
            "conv_q": obs,
            "conv_a": act,
            "connect_sym": self.connect_sym,
            "data_id": data['data_id'],
        }
 