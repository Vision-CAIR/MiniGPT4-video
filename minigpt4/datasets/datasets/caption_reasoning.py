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


from minigpt4.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset

from collections import OrderedDict


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )


# class CaptionReasonDataset(VQADataset, __DisplMixin):
class CaptionReasonDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.instruction_pool =[
            "[reasoning] {}"
        ]
        # print(ann_path)
        with open(ann_path, 'r') as f:
            self.ann = json.load(f)


        # exist_annotation = []
        # for ann in self.annotation:
        #     image_path = os.path.join(self.vis_root, ann["image"].split('/')[-1])
        #     if os.path.exists(image_path):
        #         exist_annotation.append(ann)
        # self.annotation = exist_annotation


    def get_data(self, index):
        ann = self.ann[index]

        image_path = os.path.join(self.vis_root, ann["image"].split('/')[-1])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        question_id = ann["question_id"]

        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        answer = random.choices(answers, weights=weights, k=1)[0]  # random sample an answer according to weights



        grounded_caption = ann["grounded_caption"]
        detailed_caption = ann["detailed_caption"]
        return {
            "image": image,
            "question": question,
            "question_id": question_id,
            "answer": answer,
            "detailed_caption": detailed_caption,
            "grounded_caption": grounded_caption
        }

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        data = self.get_data(index)

        question =data['question']
        detailed_caption = data["detailed_caption"]
        grounded_caption = data["grounded_caption"]

        instruction = random.choice(self.instruction_pool).format(question)
        instruction = "<Img><ImageHere></Img> {}".format(instruction)

        answer = grounded_caption+" short answer: "+data['answer']
        # print("instruction", instruction)
        # print("answer", answer)


        return {
            "image": data['image'],
            "question_id": data["question_id"],
            "instruction_input": instruction,
            "answer": answer,
        }
