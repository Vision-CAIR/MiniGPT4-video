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

from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset


def sample_object_bbox(objects, bbox):

    
    
    zipped_list = list(zip(objects, bbox))

    # Shuffle the zipped list
    random.shuffle(zipped_list)

    # Generate the new string with interleaved format
    # interleaved_list = str([{'{},{}'.format(obj, str(bbox).replace("[","").replace("]","") )} for obj, bbox in zipped_list])
    
    # print("objects", objects)
    # print("bbox",bbox)
    
    interleaved_list = str([{'{},{}'.format(obj, bbox.strip())} for obj, bbox in zipped_list]).replace("'","").replace("[","").replace("]","")

    # interleaved_list = " "+interleaved_list
    # print(interleaved_list)
    return interleaved_list

def bbox_to_object(objects, bbox):

    index_sample = random.sample(range(len(objects)),1)[0]

    sample_object = str(objects[index_sample])
    sample_bbox = bbox[index_sample]
    # sample_center_point = center_point[index_sample]

    sample_bbox = r"{"+str(sample_bbox) + "}"
    return sample_bbox, sample_object

def object_to_bbox(objects, bbox, center_point):
    index_sample = random.sample(range(len(objects)),1)[0]

    sample_object = objects[index_sample]
    sample_bbox = bbox[index_sample]
    sample_center_point = center_point[index_sample]

    instruction = "what is object and the bounding box in the center coordinate of "+str(sample_center_point)+"? "
    answer = "{"+str(sample_object)+","+str(sample_bbox)+"}"



    return instruction, answer


class LVISBBOXDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        objects = sample[1]["objects"]
        boxes = sample[1]["bbox"]


        new_bboxes = []

        image_size = sample[0].shape[1]
        image_size = 100
        for index in range(len(boxes)):
            box = boxes[index]
            x1 = int(box[0]*image_size)
            y1 = int(box[1]*image_size)
            x2 = x1 + int(box[2]*image_size)
            y2 = y1 + int(box[3]*image_size)
            assert x1>=0 and x1<=image_size
            assert x2>=0 and x2<=image_size
            assert y1>=0 and y1<=image_size
            assert y2>=0 and y2<=image_size
            
            new_bbox = " <"+str(x1)+"><"+str(y1)+"><"+str(x2)+"><"+str(y2)+">"
            # new_bbox = " <"+str(x1)+"><"+str(y1)+"><"+str(x2)+"><"+str(y2)+">"
            new_bboxes.append(new_bbox)

        instruction = r"Given an image, identify the objects and their bounding boxes in the format of {object,x1 y1 x2 y2}. "
        instruction = "<Img><ImageHere></Img> {}".format(self.text_processor(instruction))

        answer = sample_object_bbox(objects, new_bboxes)

        # print("instruction",instruction)
        # print("answer", answer)

        return {
            "image": sample[0],
            "instruction_input": instruction,
            "answer": self.text_processor(answer),
            "data_type": "bbox",
            "question_split": True
        }


class LVISBboxToObjectDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)


        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )
        # self.instruction_pool = [
        #     "###Human: <Img><ImageHere></Img> what object is in this bounding box location {}###Assistant: ",
        #     "###Human: <Img><ImageHere></Img> what object is in this location {}###Assistant: ",
        #     "###Human: <Img><ImageHere></Img> identify the object present at this location {}###Assistant: ",
        #     "###Human: <Img><ImageHere></Img> what is it in bounding box location{}###Assistant: ",
        #     "###Human: <Img><ImageHere></Img> describe this object in {} ###Assistant: ",
        #     "###Human: <Img><ImageHere></Img> this {} is ###Assistant: ",
        #     "###Human: <Img><ImageHere></Img> the object in {} is ###Assistant: ",
        #     "###Human: <Img><ImageHere></Img> please tell me what is inside the bounding box position {} ###Assistant: ",
        #     "###Human: <Img><ImageHere></Img> what can you find in the bounding box area at position {}? ###Assistant: ",
        #     "###Human: <Img><ImageHere></Img> what is the object occupying this bbox area {}###Assistant: ",
        #     "###Human: <Img><ImageHere></Img> could you identify the content within the bounding box located at {}###Assistant: ",
        # ]


        self.instruction_pool = [
            "what object is in this bounding box location {} ",
            "what object is in this location {} ",
            "identify the object present at this location {} ",
            "what is it in bounding box location{} ",
            "describe this object in {} ",
            "this {} is ",
            "the object in {} is ",
            "please tell me what is inside the bounding box position {} ",
            "what can you find in the bounding box area at position {}? ",
            "what is the object occupying this area {} ",
            "could you identify the content within the bounding box located at {} ",
            ]
    def to_dict(self, sample):
            
        objects = sample[1]["objects"]
        boxes = sample[1]["bbox"]

        new_bboxes = []

        image_size = sample[0].shape[1]
        image_size= 100
        for index in range(len(boxes)):
            box = boxes[index]
            x1 = int(box[0]*image_size)
            y1 = int(box[1]*image_size)
            x2 = x1 + int(box[2]*image_size)
            y2 = y1 + int(box[3]*image_size)
            assert x1>=0 and x1<=image_size
            assert x2>=0 and x2<=image_size
            assert y1>=0 and y1<=image_size
            assert y2>=0 and y2<=image_size
            
            new_bbox = "<"+str(x1)+"><"+str(y1)+"><"+str(x2)+"><"+str(y2)+">"
            new_bboxes.append(new_bbox)
        
        bbox, object = bbox_to_object(objects, new_bboxes)
        instruction = random.choice(self.instruction_pool).format(bbox)

        # instruction = "###Human: <Img><ImageHere></Img> {} ###Assistant: ".format(instruction)

        instruction = " <Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": sample[0],
            "instruction_input": instruction,
            "answer": self.text_processor(object),
            "data_type": "bbox",
            "question_split": True
        }


