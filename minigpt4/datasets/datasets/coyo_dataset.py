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
from minigpt4.datasets.datasets.base_dataset import BaseDataset


class COYOCaptionWDSDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json"),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )
  
        self.instruction_pool = [
            '[grounding] Briefly describe this image with grounding objects.',
            '[grounding] Provide a concise depiction of this image with grounding objects.',
            '[grounding] Present a short description of this image with grounding objects.',
            '[grounding] Summarize this image in a few words with grounding objects.',
            '[grounding] A short image caption with grounding objects:',
            '[grounding] A short image description with grounding objects:',
            '[grounding] Write a short description for the image with grounding objects.',
            '[grounding] Write a description for the photo with grounding objects.',
            '[grounding] Briefly describe the content of the image with grounding objects.',
            '[grounding] Please provide a short depiction of the picture with grounding objects.',
        ]

        # self.instruction_pool = [
        #     '[grounding] Briefly describe this image.',
        #     '[grounding] Provide a concise depiction of this image.',
        #     '[grounding] Present a short description of this image.',
        #     '[grounding] Summarize this image in a few words.',
        #     '[grounding] A short image caption:',
        #     '[grounding] A short image description:',
        #     '[grounding] A photo of',
        #     '[grounding] An image that shows',
        #     '[grounding] Write a short description for the image.',
        #     '[grounding] Write a description for the photo.',
        #     '[grounding] Provide a description of what is presented in the photo.',
        #     '[grounding] Briefly describe the content of the image.',
        #     '[grounding] Can you briefly explain what you see in the image?',
        #     '[grounding] Could you use a few words to describe what you perceive in the photo?',
        #     '[grounding] Please provide a short depiction of the picture.',
        #     '[grounding] Using language, provide a short account of the image.',
        #     '[grounding] Use a few words to illustrate what is happening in the picture.',
        # ]

    def generate_ground_caption(self,image_caption, phrases, bounding_boxes):
        
        grounded_caption = image_caption
        
        # Iterate over the phrases and bounding boxes
        phrase_bbox={}
        for phrase, bbox in zip(phrases, bounding_boxes):
            # Replace the phrase with the grounded HTML format
            # print(phrase, bbox, type(phrase), type(bbox))

            if phrase not in phrase_bbox.keys():
                grounded_phrase = "<p>{}</p> ".format(phrase)
                grounded_phrase_bbox = grounded_phrase+str(bbox)
            else:
                grounded_phrase = phrase_bbox[phrase]

                grounded_phrase_bbox = grounded_phrase+"<delim>"+str(bbox)

            phrase_bbox[phrase] = grounded_phrase_bbox
            

        grounded_caption = grounded_caption.replace(phrase, grounded_phrase_bbox)
        
        return grounded_caption


    def preprocess_ground_caption(self, sample):

        # info = self.ann["data"][index]
        image_id = sample[1]["id"]


        caption = sample[1]["caption"]
        ref_exps = sample[1]["noun_chunks"]
        image_size = 100

        bboxs = []
        ref_phrases = []
        for item in ref_exps:
            phrase_start = int(item[0])
            phrase_end = int(item[1])

            x_min = item[2]
            y_min = item[3]
            x_max = item[4]
            y_max = item[5]
            ref_phrase = caption[phrase_start: phrase_end]

            x1 = int(x_min*image_size)
            y1 = int(y_min*image_size)
            x2 = int(x_max*image_size)
            y2 = int(y_max*image_size)
            assert x1>=0 and x1<=image_size
            assert x2>=0 and x2<=image_size
            assert y1>=0 and y1<=image_size
            assert y2>=0 and y2<=image_size
            # print(x1, y2, x2, y2)
            bbox = [str(x1),str(y1),str(x2),str(y2)]
            # bbox = "<"+str(x1)+"><"+str(y1)+"><"+str(x2)+"><"+str(y2)+">"
            bbox = "{{<{}><{}><{}><{}>}}".format(*bbox)
            bboxs.append(bbox)
            ref_phrases.append(ref_phrase)
        
        grounded_caption = self.generate_ground_caption(caption, ref_phrases,bboxs)
       


        return {
            "answer": grounded_caption
        }


    def to_dict(self, sample):
        data = self.preprocess_ground_caption(sample)

        instruction = random.choice(self.instruction_pool)
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        answer = self.text_processor(data['answer'])
        return {
            "image": sample[0],
            "instruction_input": instruction,
            "answer": answer,
        }



class COYOBoxToPhraseWDSDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )


        self.instruction_pool = [
            "[identify] {}",
            "[identify] what object is in this location {}",
            "[identify] identify the object present at this location {}",
            "[identify] what is it in {}",
            "[identify] describe this object in {}",
            "[identify] this {} is",
            "[identify] the object in {} is",
            ]
    def bbox_phrase_preprocess(self, sample):

        caption = sample[1]["caption"]
        # ref_exps = sample[1]["ref_exps"]
        ref_exps = sample[1]["noun_chunks"]
        image_size = 100

        bboxs = []
        ref_phrases = []
        for item in ref_exps:
            # print(item)
            phrase_start = int(item[0])
            phrase_end = int(item[1])

            x_min = item[2]
            y_min = item[3]
            x_max = item[4]
            y_max = item[5]
            ref_phrase = caption[phrase_start: phrase_end]

            x1 = int(x_min*image_size)
            y1 = int(y_min*image_size)
            x2 = int(x_max*image_size)
            y2 = int(y_max*image_size)
            assert x1>=0 and x1<=image_size
            assert x2>=0 and x2<=image_size
            assert y1>=0 and y1<=image_size
            assert y2>=0 and y2<=image_size

            bbox = [str(x1),str(y1),str(x2),str(y2)]

            
            # bbox = "<"+str(x1)+"><"+str(y1)+"><"+str(x2)+"><"+str(y2)+">"
            bbox = "{{<{}><{}><{}><{}>}}".format(*bbox)
            bboxs.append(bbox)
            ref_phrases.append(ref_phrase)

            # print(ref_phrase, bbox)

        index = random.randint(0, len(bboxs)-1)

        # Retrieve the corresponding elements
        sampled_bbox = bboxs[index]
        sampled_phrase = ref_phrases[index]

        return {
            "instruction_input": sampled_bbox,
            "answer": sampled_phrase,
        }

    def to_dict(self, sample):

        data = self.bbox_phrase_preprocess(sample)

        instruction = random.choice(self.instruction_pool).format(data['instruction_input'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        answer = self.text_processor(data['answer'])

        return {
            "image": sample[0],
            "instruction_input": instruction,
            "answer": answer,
        }



class COYOPhraseToBoxWDSDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

        self.instruction_pool = [
            "[refer] {}",
            "[refer] give me the location of {}",
            "[refer] where is {} ?",
            "[refer] from this image, tell me the location of {}",
            "[refer] the location of {} is ",
            "[refer] could you tell me the location for {}?",
            "[refer] where can I locate the {}?",
        ]

        # self.instruction_pool = [
        #     # "[refer] {}",
        #     "[refer] give me the bounding box location of {}",
        #     "[refer] where is bounding box location of {} ?",
        #     "[refer] from this image, tell me the bounding box location of {}",
        #     "[refer] the bounding box location of {} is",
        #     "[refer] could you tell me the bounding box location for {} ?",
        #     "[refer] where can I locate the bounding box of {} ?",
        # ]
    def phrase_bbox_preprocess(self, sample):

        caption = sample[1]["caption"]
        ref_exps = sample[1]["ref_exps"]
        image_size = 100

        bboxs = []
        ref_phrases = []
        for item in ref_exps:
            phrase_start = int(item[0])
            phrase_end = int(item[1])

            x_min = item[2]
            y_min = item[3]
            x_max = item[4]
            y_max = item[5]
            ref_phrase = caption[phrase_start: phrase_end]

            x1 = int(x_min*image_size)
            y1 = int(y_min*image_size)
            x2 = int(x_max*image_size)
            y2 = int(y_max*image_size)
            assert x1>=0 and x1<=image_size
            assert x2>=0 and x2<=image_size
            assert y1>=0 and y1<=image_size
            assert y2>=0 and y2<=image_size
            
            # bbox = "<"+str(x1)+"><"+str(y1)+"><"+str(x2)+"><"+str(y2)+">"
            bbox = [str(x1),str(y1),str(x2),str(y2)]
            
            bbox = "{{<{}><{}><{}><{}>}}".format(*bbox)
            bboxs.append(bbox)
            ref_phrases.append(ref_phrase)

        index = random.randint(0, len(bboxs)-1)

        # Retrieve the corresponding elements
        sampled_bbox = bboxs[index]
        sampled_phrase = ref_phrases[index]

        return {
            "instruction_input": sampled_phrase,
            "answer": sampled_bbox,
        }


    def to_dict(self, sample):
        data = self.phrase_bbox_preprocess(sample)
        instruction_input = self.text_processor(data['instruction_input'])
        instruction = random.choice(self.instruction_pool).format(instruction_input)
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": sample[0],
            "instruction_input": instruction,
            "answer": data["answer"],
        }




# class COYOBBoxPhraseDataset(Dataset):
#     def __init__(self, vis_processor, text_processor, vis_root, ann_path):
#         """
#         vis_root (string): Root directory of images (e.g. coco/images/)
#         ann_root (string): directory to store the annotation file
#         """
#         self.vis_root = vis_root

#         self.vis_processor = vis_processor
#         self.text_processor = text_processor

#         self.ann = {"data":[]}

        
#         with open(ann_path, 'r') as f:
#             for line in f.readlines():
#                 line = line.strip()
#                 # print(line, type(line))
#                 try:
#                     item = json.loads(line.strip())
#                 except:
#                     print(line)
#                     # print(item)
#                     assert False

#                 # print(item, type(item))
#                 # assert False
#                 self.ann["data"].append(item)


#         self.bbox_phrase_instruction_pool = [
#             "<Img><ImageHere></Img> what object is in this bounding box location {} ",
#             "<Img><ImageHere></Img> what object is in this location {} ",
#             "<Img><ImageHere></Img> identify the object present at this location {} ",
#             "<Img><ImageHere></Img> what is it in bounding box location{} ",
#             "<Img><ImageHere></Img> describe this object in {} ",
#             "<Img><ImageHere></Img> this {} is ",
#             "<Img><ImageHere></Img> the object in {} is ",
#             "<Img><ImageHere></Img> please tell me what is inside the bounding box position {} ",
#             "<Img><ImageHere></Img> what can you find in the bounding box area at position {}? ",
#             "<Img><ImageHere></Img> what is the object occupying this area {} ",
#             "<Img><ImageHere></Img> could you identify the content within the bounding box located at {} ",
#             ]

#     def __len__(self):
#         return len(self.ann["data"])

#     def bbox_phrase_preprocess(self, index):

#         info = self.ann["data"][index]
#         image_id = info["id"]

#         image_file = str(image_id)+".jpg"
#         image_path = os.path.join(self.vis_root, image_file)
#         image = Image.open(image_path).convert("RGB")
#         image = self.vis_processor(image)

#         caption = info["caption"]
#         ref_exps = info["ref_exps"]
#         image_size = 100

#         bboxs = []
#         ref_phrases = []
#         for item in ref_exps:
#             # print(item)
#             phrase_start = int(item[0])
#             phrase_end = int(item[1])

#             x_min = item[2]
#             y_min = item[3]
#             x_max = item[4]
#             y_max = item[5]
#             ref_phrase = caption[phrase_start: phrase_end]

#             x1 = int(x_min*image_size)
#             y1 = int(y_min*image_size)
#             x2 = int(x_max*image_size)
#             y2 = int(y_max*image_size)
#             assert x1>=0 and x1<=image_size
#             assert x2>=0 and x2<=image_size
#             assert y1>=0 and y1<=image_size
#             assert y2>=0 and y2<=image_size

#             bbox = [str(x1),str(y1),str(x2),str(y2)]

            
#             # bbox = "<"+str(x1)+"><"+str(y1)+"><"+str(x2)+"><"+str(y2)+">"
#             bbox = "{{<{}><{}><{}><{}>}}".format(*bbox)
#             bboxs.append(bbox)
#             ref_phrases.append(ref_phrase)

#             # print(ref_phrase, bbox)

#         index = random.randint(0, len(bboxs)-1)

#         # Retrieve the corresponding elements
#         sampled_bbox = bboxs[index]
#         sampled_phrase = ref_phrases[index]

#         return {
#             "image": image,
#             "instruction_input": sampled_phrase,
#             "answer": sampled_bbox,
#             "image_id": info['id'],
#         }



#     def __getitem__(self, index):

#         data = self.preprocess(index)
#         instruction = random.choice(self.instruction_pool).format(data['instruction_input'])
#         return {
#             "image": data['image'],
#             "instruction_input": instruction,
#             "answer": data['answer'],
#             "image_id": data['image_id'],
#         }
