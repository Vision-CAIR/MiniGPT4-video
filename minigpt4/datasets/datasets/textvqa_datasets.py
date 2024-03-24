"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch

from PIL import Image

from minigpt4.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset

from collections import OrderedDict


# class textVQADataset(VQADataset):
#     def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
#         super().__init__(vis_processor, text_processor, vis_root, ann_paths)

#     def collater(self, samples):
#         image_list, question_list, answer_list, weight_list = [], [], [], []
    
#         num_answers = []
    
#         for sample in samples:
#             image_list.append(sample["image"])
#             question_list.append(sample["text_input"])
    
#             weight_list.extend(sample["weights"])
    
#             answers = sample["answers"]
    
#             answer_list.extend(answers)
#             num_answers.append(len(answers))
    
#         return {
#             "image": torch.stack(image_list, dim=0),
#             "text_input": question_list,
#             "answer": answer_list,
#             "weight": torch.Tensor(weight_list),
#             "n_answers": torch.LongTensor(num_answers),
#         }



from minigpt4.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset

class textVQAEvalDataset(VQADataset):
    def __init__(self, vis_processor, text_processor, vis_root=None, ann_paths=None):
#         super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        from datasets import load_dataset
        self.annotation = load_dataset("textvqa", split="validation")

    def __getitem__(self, index):
        ann = self.annotation[index]
        image = ann["image"].convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        
        instruction = random.choice(self.instruction_pool).format(question)
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)
        print("instruction", instruction)
        answers = ann["answers"]

        if "unk" in answers:
            print(answers)
        return {
            "image": image,
            "text_input": question,
            "answer": answers,
            # 'image_path': image_path,
            "instruction_input": instruction,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }

    
dataset = textVQAEvalDataset(vis_processor, text_processor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)