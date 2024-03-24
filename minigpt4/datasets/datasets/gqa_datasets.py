"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json

from PIL import Image

from minigpt4.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset

from collections import OrderedDict
import random

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


class GQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.instruction_pool =[
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        instruction = random.choice(self.instruction_pool).format(question)
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        answers = self.text_processor(ann["answer"])
        if "unk" in answers:
            print("gqa",answers)

        # print(answers)

        return {
            "image": image,
            "instruction_input": instruction,
            "answer": answers,
            # "weights": weights,
        }


class GQAEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. gqa/images/)
        ann_root (string): directory to store the annotation file
        """
        
        self.instruction_pool = [
#             '{}',
#             'Question: {}',
#             '{} A short answer to the question is',
#             'Q: {} A:',
            # '[vqa] Question: {} Short answer:',
            "[vqa] Based on the image, respond to this question with a short answer: {}"
#             'Given the image, answer the following question with no more than three words. {}',
#             'Based on the image, respond to this question with a short answer: {}.',
#             'Use the provided image to answer the question: {} Provide your answer as short as possible.',
#             'What is the answer to the following question? "{}"',
#             'The question "{}" can be answered using the image. A short answer is'
        ]

        self.vis_root = vis_root

        self.annotation = json.load(open(ann_paths[0]))

        ## TODO: support inference method == 'ranking'
        answer_list_path = ann_paths[1] if len(ann_paths) > 1 else ''
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            self.answer_list = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        
        instruction = random.choice(self.instruction_pool).format(question)
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)
        
        if "answer" in ann:
            # answer is a string
            answer = ann["answer"]
        else:
            answer = None

        return {
            "image": image,
            "text_input": question,
            "answer": answer,
            'image_path': image_path,
            "instruction_input": instruction,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }
