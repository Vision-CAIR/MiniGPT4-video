import os
import json
import pickle
import math
import random
import glob

import numpy as np
import torch
import time
import cv2

from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import cv2
from pycocotools.coco import COCO

from minigpt4.datasets.datasets.base_dataset import BaseDataset


def pt_paint(strokes, num_steps=999):
    # Create a black canvas
    img = Image.new('RGB', (256, 256), color='black')
    draw = ImageDraw.Draw(img)
    max_steps = len(strokes)
    num_steps = min(num_steps, max_steps)

    for i in range(0, num_steps):
        stroke = strokes[i]

        x = stroke[0]
        y = stroke[1]
        w = stroke[2]
        h = stroke[3]
        theta = stroke[4] * 180
        rgb = tuple(int(val * 255) for val in stroke[5:8])  # Scale RGB values to 0-255

        # Convert degrees to radians for rotation
        angle_rad = theta * (3.141592653589793 / 180.0)
        cos_val = math.cos(angle_rad)
        sin_val = math.sin(angle_rad)

        # Calculate the coordinates of the rectangle vertices after rotation
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y - h/2
        x3 = x + w/2
        y3 = y + h/2
        x4 = x - w/2
        y4 = y + h/2

        # Rotate the rectangle coordinates
        x1_new = cos_val * (x1 - x) - sin_val * (y1 - y) + x
        y1_new = sin_val * (x1 - x) + cos_val * (y1 - y) + y
        x2_new = cos_val * (x2 - x) - sin_val * (y2 - y) + x
        y2_new = sin_val * (x2 - x) + cos_val * (y2 - y) + y
        x3_new = cos_val * (x3 - x) - sin_val * (y3 - y) + x
        y3_new = sin_val * (x3 - x) + cos_val * (y3 - y) + y
        x4_new = cos_val * (x4 - x) - sin_val * (y4 - y) + x
        y4_new = sin_val * (x4 - x) + cos_val * (y4 - y) + y

        # Draw the rotated rectangle
        draw.polygon([(x1_new, y1_new), (x2_new, y2_new), (x3_new, y3_new), (x4_new, y4_new)], fill=rgb)

    return img


def pt_stroke2str(single_stroke):
    x, y, w, h, theta, r, g, b = single_stroke
    theta = theta * 180
    r, g, b = r * 255, g * 255, b * 255
    param = [x, y, w, h, theta, r, g, b]
    param = ','.join([str(int(i)) for i in param])

    str_stroke = '({})'.format(param)
    return str_stroke


class PaintPTCOCODataset(Dataset):
    def __init__(self, vis_processor, text_processor, img_root, stroke_root, max_step=200):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.img_root = img_root
        self.stroke_root = stroke_root
        self.image_ids = [file.split('/')[-1].split('.')[0]
                          for file in glob.glob(os.path.join(self.stroke_root, '*.pkl'))]
        self.max_step = max_step
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def __len__(self):
        return len(self.image_ids)

    def preprocess(self, index, step=-1):
        image_id = self.image_ids[index]
        with open(os.path.join(self.stroke_root, '{}.pkl'.format(image_id)), "rb") as f:
            strokes_dict = pickle.load(f)

        strokes = np.concatenate(strokes_dict['strokes'], axis=0)
        if step < 0:
            step = random.randint(0, min(len(strokes) - 1, self.max_step))
        canvas = pt_paint(strokes, num_steps=step)
        next_stroke = strokes[step]

        image_file = '{}.jpg'.format(image_id)
        image_path = os.path.join(self.img_root, image_file)
        orig_image = Image.open(image_path).convert("RGB")

        return {
            "orig_image": orig_image,
            "canvas": canvas,
            "next_stroke": pt_stroke2str(next_stroke),
            "image_id": image_id,
        }

    def __getitem__(self, index):
        data = self.preprocess(index)
        orig_image = self.vis_processor(data['orig_image'])
        canvas = self.vis_processor(data['canvas'])
        instruction = "<Image><ImageHere><Canvas><ImageHere> Next Stroke: "

        return {
            "image": torch.stack([orig_image, canvas], dim=0),
            "instruction_input": instruction,
            "answer": data['next_stroke'],
            "image_id": data['image_id'],
            "length": 2
        }


def normal(x, width):
    return (int)(x * (width - 1) + 0.5)


def draw(f, canvas=None, width=128, res=100):
    x0, y0, x1, y1, x2, y2, z0, z2, w0, w2, b, g, r = [float(i) for i in f]
    x1 = x0 + (x2 - x0) * x1
    y1 = y0 + (y2 - y0) * y1
    x0 = normal(x0, width)
    x1 = normal(x1, width)
    x2 = normal(x2, width)
    y0 = normal(y0, width)
    y1 = normal(y1, width)
    y2 = normal(y2, width)
    z0 = (int)(1 + z0 * width // 4)
    z2 = (int)(1 + z2 * width // 4)
    if canvas is None:
        canvas = np.zeros([width, width, 4])
    tmp = 1. / res
    for i in range(res):
        t = i * tmp
        x = (int)((1-t) * (1-t) * x0 + 2 * t * (1-t) * x1 + t * t * x2)
        y = (int)((1-t) * (1-t) * y0 + 2 * t * (1-t) * y1 + t * t * y2)
        z = (int)((1-t) * z0 + t * z2)
        # w = (1-t) * w0 + t * w2
        w = 1

        cv2.circle(canvas, (y, x), z, [w, r * w, g * w, b * w], -1)

    return canvas


def rl_decode(x, canvas, res=100):
    stroke = []
    color_stroke = []
    for step in range(x.shape[1]):
        stroke_canvas = np.zeros([canvas.shape[-1], canvas.shape[-1], 4], dtype=np.float32)  # alpha, alpha * r, alpha * g, alpha * b
        for idx in range(x.shape[0]):
            stroke_canvas = draw(x[idx, step], canvas=stroke_canvas, width=canvas.shape[-1], res=res)
        stroke_canvas = stroke_canvas.transpose(2, 0, 1)
        stroke.append(stroke_canvas[:1])
        color_stroke.append(stroke_canvas[1:])

    for i in range(len(stroke)):
        canvas = canvas * (1 - stroke[i]) + color_stroke[i]
    return canvas


def rel2abs(strokes, n_d=4):
    abs_strokes = []
    for i, stroke in enumerate(strokes):
        yi = i % n_d
        xi = i // n_d
        stroke = np.stack([
            stroke[:, 0] / n_d + xi / n_d,
            stroke[:, 1] / n_d + yi / n_d,
            stroke[:, 2] / n_d + xi / n_d,
            stroke[:, 3] / n_d + yi / n_d,
            stroke[:, 4] / n_d + xi / n_d,
            stroke[:, 5] / n_d + yi / n_d,
            stroke[:, 6] / n_d,
            stroke[:, 7] / n_d,
            stroke[:, 8],
            stroke[:, 9],
            stroke[:, 10],
            stroke[:, 11],
            stroke[:, 12],
        ], axis=1)
        abs_strokes.append(stroke)
    abs_strokes = np.stack(abs_strokes)
    return abs_strokes


def rl_paint(strokes_dict, step, width=256, single_stroke=False):
    canvas = np.zeros([1, 3, width, width], dtype=np.float32)

    if_fine_strokes = [int(len(strokes.shape) > 2) for strokes in strokes_dict['strokes']]
    if single_stroke:
        n_steps = (len(if_fine_strokes) - sum(if_fine_strokes)) * 5 + 16 * 5 * sum(if_fine_strokes)
    else:
        n_steps = len(if_fine_strokes) + 4 * sum(if_fine_strokes)

    step = min(step, n_steps-1)

    for strokes in strokes_dict['strokes']:

        strokes = strokes.astype(np.float32)
        if len(strokes.shape) < 3:  # coarse stage. shape 5, 13
            if single_stroke:  # 1 stroke per step
                actions_list = [stroke[None, None] for stroke in strokes]
            else:  # 5 strokes per step
                actions_list = [strokes[None]]
        else:  # fine stage. shape 16, 5, 13
            strokes = rel2abs(strokes)

            if single_stroke:  # 1 stroke per step
                strokes = strokes.transpose(1, 0, 2)
                actions_list = [stroke[None, None] for step_strokes in strokes for stroke in step_strokes]

            else:  # 16 strokes per step. each variable strokes contains 5 steps
                actions_list = [strokes[:, i:i+1] for i in range(strokes.shape[1])]

        for actions in actions_list:
            if step > 0:
                canvas = rl_decode(actions, canvas, res=100)
                step = step - 1
            else:
                next_stroke = actions
                return canvas, next_stroke

    raise StopIteration


def rl_stroke2str(action):
    a, b, _ = action.shape

    if a == 1 and b == 5: # coarse step, contains 5 strokes
        action = action[0]  # 5 x 13
        tag = '[coarse]'
    elif a == 16 and b == 1: # fine step. contains 16 strokes
        action = action[:, 0]  # 16 x 13
        tag = '[detail]'
    elif a == 1 and b == 1:
        action = action[0]
        tag = ''
    else:
        raise ValueError

    strokes = []
    for i, stroke in enumerate(action):
        stroke = [str(int(i * 255)) for i in stroke]
        stroke = ",".join(stroke)
        stroke = "{}({})".format(i, stroke)
        strokes.append(stroke)
    strokes = ';'.join(strokes)
    strokes = tag + strokes

    return strokes


def rlo_stroke2str(action):
    a, b, _ = action.shape

    if a == 1 and b == 5: # coarse step, contains 5 strokes
        action = action[0]  # 5 x 13
        tag = '[coarse]'
    elif a == 16 and b == 1: # fine step. contains 16 strokes
        action = action[:, 0]  # 16 x 13
        tag = '[detail]'
    elif a == 1 and b == 1:
        action = action[0]
        tag = ''
    else:
        raise ValueError

    strokes = []

    for i, stroke in enumerate(action):
        x0, y0, x1, y1, x2, y2, z0, z2, w0, w2, b, g, r = stroke
        stroke = [x0, y0, x1, y1, x2, y2, z0, z2, b, g, r]  # remove unused transparancy
        stroke = [str(int(i * 255)) for i in stroke]
        stroke = ",".join(stroke)
        stroke = "{}({})".format(i, stroke)
        strokes.append(stroke)
    strokes = ';'.join(strokes)
    strokes = tag + strokes

    return strokes


class PaintRLCOCODataset(Dataset):
    def __init__(self, vis_processor, text_processor, img_root, stroke_root, single_stroke=False, max_step=50):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.img_root = img_root
        self.stroke_root = stroke_root
        self.image_ids = [file.split('/')[-1].split('.')[0]
                          for file in glob.glob(os.path.join(self.stroke_root, '*.pkl'))]
        self.max_step = max_step
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.single_stroke=single_stroke
        self.width = 256

    def __len__(self):
        return len(self.image_ids)

    def preprocess(self, index, step=-1):
        image_id = self.image_ids[index]
        image_file = '{}.jpg'.format(image_id)
        image_path = os.path.join(self.img_root, image_file)
        orig_image = Image.open(image_path).convert("RGB")

        with open(os.path.join(self.stroke_root, '{}.pkl'.format(image_id)), "rb") as f:
            strokes_dict = pickle.load(f)

        if_fine_strokes = [int(len(strokes.shape) > 2) for strokes in strokes_dict['strokes']]
        if self.single_stroke:
            n_steps = (len(if_fine_strokes) - sum(if_fine_strokes)) * 5 + 16 * 5 * sum(if_fine_strokes)
        else:
            n_steps = len(if_fine_strokes) + 4 * sum(if_fine_strokes)

        if step < 0:
            step = random.randint(0, min(n_steps - 1, self.max_step))

        canvas, next_stroke = rl_paint(strokes_dict, step, width=self.width, single_stroke=self.single_stroke)
        canvas = Image.fromarray((canvas[0].transpose(1, 2, 0) * 255).astype(np.uint8))

        return {
            "orig_image": orig_image,
            "canvas": canvas,
            "next_stroke": rl_stroke2str(next_stroke),
            "image_id": image_id,
        }

    def __getitem__(self, index):
        data = self.preprocess(index)
        orig_image = self.vis_processor(data['orig_image'])
        canvas = self.vis_processor(data['canvas'])
        instruction = "<Image><ImageHere><Canvas><ImageHere> Action: "

        return {
            "image": torch.stack([orig_image, canvas], dim=0),
            "instruction_input": instruction,
            "answer": data['next_stroke'],
            "image_id": data['image_id'],
            "length": 2
        }


class PaintLanRLOpaqueCOCODataset(Dataset):
    def __init__(self, vis_processor, text_processor, img_root, stroke_root, ann_path, single_stroke=False, max_step=50):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.img_root = img_root
        self.stroke_root = stroke_root
        self.image_ids = [file.split('/')[-1].split('.')[0]
                          for file in glob.glob(os.path.join(self.stroke_root, '*.pkl'))]
        self.max_step = max_step
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.single_stroke = single_stroke

        self.captions = {}
        with open(ann_path, 'r') as f:
            anns = json.load(f)
        for ann in anns['annotations']:
            if ann['image_id'] in self.captions:
                self.captions[ann['image_id']].append(ann['caption'])
            else:
                self.captions[ann['image_id']] = [ann['caption']]
        for idx in self.image_ids:
            assert int(idx) in self.captions

        self.width = 256
        self.instruction = "Task: {}\nCanvas: <ImageHere> Action: "

    def __len__(self):
        return len(self.image_ids)

    def preprocess(self, index, step=-1):
        image_id = self.image_ids[index]
        image_file = '{}.jpg'.format(image_id)
        image_path = os.path.join(self.img_root, image_file)
        orig_image = Image.open(image_path).convert("RGB")
        captions = self.captions[int(image_id)]

        with open(os.path.join(self.stroke_root, '{}.pkl'.format(image_id)), "rb") as f:
            strokes_dict = pickle.load(f)

        if_fine_strokes = [int(len(strokes.shape) > 2) for strokes in strokes_dict['strokes']]
        if self.single_stroke:
            n_steps = (len(if_fine_strokes) - sum(if_fine_strokes)) * 5 + 16 * 5 * sum(if_fine_strokes)
        else:
            n_steps = len(if_fine_strokes) + 4 * sum(if_fine_strokes)

        if step < 0:
            step = random.randint(0, min(n_steps - 1, self.max_step))

        canvas, next_stroke = rl_paint(strokes_dict, step, width=self.width, single_stroke=self.single_stroke)
        canvas = Image.fromarray((canvas[0].transpose(1, 2, 0) * 255).astype(np.uint8))

        return {
            "orig_image": orig_image,
            "captions": captions,
            "canvas": canvas,
            "next_stroke": rlo_stroke2str(next_stroke),
            "image_id": image_id,
        }

    def __getitem__(self, index):
        data = self.preprocess(index)
        canvas = self.vis_processor(data['canvas'])
        instruction = self.instruction.format(random.choice(data['captions']))

        return {
            "image": canvas,
            "instruction_input": instruction,
            "answer": data['next_stroke'],
            "image_id": data['image_id'],
        }


class PaintPixelCOCODataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, res):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.res = res
        self.img_ids = {}
        n = 0

        self.filter_anntation = []

        for ann in self.annotation:
            if "train" in ann["image"]:
                self.filter_anntation.append(ann)
        self.annotation = self.filter_anntation

        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):
        ann = self.annotation[index]

        img_file = ann["image"].split("/")[-1]
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")

        pixelized = np.array(image.resize([self.res, self.res]))

        image = self.vis_processor(image)

        loc_y = random.randint(0, self.res - 1)
        loc_x = random.randint(0, self.res - 1)
        rgb = pixelized[loc_y, loc_x]

        instruction = "<Img><ImageHere></Img> [reconstruct] loc: [{},{}] rgb: ".format(loc_y, loc_x)
        answer = '[{},{},{}]'.format(rgb[0], rgb[1], rgb[2])

        return {
            "image": image,
            "answer": answer,
            "instruction_input": instruction,
        }


class SegReferCOCODataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path, res, dataset='refcoco', splitBy='unc'):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_path (string): directory to store the annotation file
        """
        self.vis_root = vis_root
        self.ann_path = ann_path
        self.splitBy = splitBy
        self.res = res

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.ann_dir = os.path.join(ann_path, dataset)
        ref_file = os.path.join(self.ann_dir, 'refs(' + splitBy + ').p')

        self.data = {}
        with open(ref_file, 'rb') as f:
            data_refs = pickle.load(f)
        data_refs = [ref for ref in data_refs if ref['split'] == 'train']  # only use train split

        for ref in data_refs:
            if ref['image_id'] in self.data:
                self.data[ref['image_id']].append(ref)
            else:
                self.data[ref['image_id']] = [ref]
        self.img_id_list = list(self.data.keys())

        # load annotations from data/dataset/instances.json
        instances_file = os.path.join(self.ann_dir, 'instances.json')
        self.coco = COCO(instances_file)

    def __len__(self):
        return len(self.img_id_list)

    def prepare_data(self, index):
        image_id = self.img_id_list[index]
        raw_anns = self.data[image_id]
        anns = []
        for ann in raw_anns:
            refers = [sentence['sent'] for sentence in ann['sentences']]
            ann_id = ann['ann_id']
            annotations = self.coco.loadAnns([ann_id])
            mask = Image.fromarray(self.coco.annToMask(annotations[0]))
            anns.append({'refers': refers, 'mask': mask})

        img_data = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.vis_root, img_data['file_name'])
        image = Image.open(image_path).convert("RGB")

        return {
            'image': image,
            'anns': anns,
        }

    def __getitem__(self, index):
        data = self.prepare_data(index)
        image = self.vis_processor(data['image'])
        all_masks = [np.array(ann['mask'].resize([self.res, self.res], 0)) for ann in data['anns']]
        ann_id = random.randint(0, len(data['anns']) - 1)

        selected_ann = data['anns'][ann_id]
        selected_refer = random.choice(selected_ann['refers'])
        pixelized_mask = all_masks[ann_id]
        all_mask = sum(all_masks)

        pixelized_mask[pixelized_mask != 0] = 1
        all_mask[all_mask != 0] = 1

        has_other_obj = bool((all_mask != pixelized_mask).sum())

        if (pixelized_mask == 0).sum() in [0, pixelized_mask.size]:  # all black or all white
            loc_y = random.randint(0, self.res - 1)
            loc_x = random.randint(0, self.res - 1)
        else:
            if random.uniform(0, 1) < 0.4:  # in 40% cases we sample object region
                # object region
                ys, xs = np.where(pixelized_mask != 0)
            else:
                # background
                dice = random.uniform(0, 1)
                if dice < 0.1:
                    # easy background points
                    ys, xs = np.where(pixelized_mask == 0)
                elif has_other_obj and dice < 0.6:
                    # points on other unrelated objects
                    other_obj_mask = cv2.bitwise_xor(pixelized_mask, all_mask)
                    ys, xs = np.where(other_obj_mask != 0)
                else:
                    # contour points around the object
                    dilate_mask = cv2.dilate(pixelized_mask, np.ones([self.res // 8, self.res // 8], dtype=np.uint8),
                                             iterations=1)
                    contour_mask = cv2.bitwise_xor(pixelized_mask, dilate_mask)
                    ys, xs = np.where(contour_mask != 0)

            idx = random.randint(0, len(ys) - 1)
            loc_y, loc_x = ys[idx], xs[idx]

        mask_value = pixelized_mask[loc_y, loc_x]

        instruction = "<Img><ImageHere></Img> [segmentation] {} loc: [{},{}] mask: ".format(
            selected_refer, loc_y, loc_x)
        answer = str(mask_value)

        return {
            "image": image,
            "answer": answer,
            "instruction_input": instruction,
        }
