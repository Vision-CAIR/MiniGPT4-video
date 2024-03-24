"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from minigpt4.common.registry import registry
from minigpt4.datasets.datasets.aok_vqa_datasets import AOKVQADataset
from minigpt4.datasets.datasets.aok_vqa_reasoning_datasets import AOKVQAReasoningDataset
#, AOKVQGDataset, AOKVQAEvalDataset
from minigpt4.datasets.datasets.coco_vqa_datasets import COCOVQADataset, COCOVQGDataset, COCOVQAEvalDataset
# from minigpt4.datasets.datasets.vg_vqa_datasets import VGVQADataset
from minigpt4.datasets.datasets.gqa_datasets import GQADataset, GQAEvalDataset
from minigpt4.datasets.datasets.doc_dataset import SingleSlideVQADataset, OCRVQADataset



@registry.register_builder("coco_vqa")
class COCOVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset
    eval_dataset_cls = COCOVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_vqa.yaml",
        "eval": "configs/datasets/coco/eval_vqa.yaml",
    }
    

# @registry.register_builder("vg_vqa")
# class VGVQABuilder(BaseDatasetBuilder):
#     train_dataset_cls = VGVQADataset
#     DATASET_CONFIG_DICT = {"default": "configs/datasets/vg/defaults_vqa.yaml"}


@registry.register_builder("ok_vqa")
class OKVQABuilder(COCOVQABuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/okvqa/defaults.yaml",
    }


@registry.register_builder("aok_vqa")
class AOKVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQADataset
    # eval_dataset_cls = AOKVQAEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/aokvqa/defaults.yaml"}

@registry.register_builder("aok_vqa_reasoning")
class AOKVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQAReasoningDataset
    # eval_dataset_cls = AOKVQAEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/aokvqa_reasoning/defaults.yaml"}


@registry.register_builder("gqa")
class GQABuilder(BaseDatasetBuilder):
    train_dataset_cls = GQADataset
    # eval_dataset_cls = GQAEvalDataset

    DATASET_CONFIG_DICT = {
        # "default": "configs/datasets/gqa/defaults.yaml",
        # "balanced_val": "configs/datasets/gqa/balanced_val.yaml",
        "default": "configs/datasets/gqa/balanced_val.yaml",
        # "balanced_testdev": "configs/datasets/gqa/balanced_testdev.yaml",
    }



@registry.register_builder("coco_vqg")
class COCOVQGBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQGDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_vqg.yaml",
    }


@registry.register_builder("ok_vqg")
class OKVQGBuilder(COCOVQGBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/okvqa/defaults_vqg.yaml",
    }


# @registry.register_builder("aok_vqg")
# class AOKVQGBuilder(BaseDatasetBuilder):
#     train_dataset_cls = AOKVQGDataset

#     DATASET_CONFIG_DICT = {"default": "configs/datasets/aokvqa/defaults_vqg.yaml"}


class DocumentVQABuilder(BaseDatasetBuilder):
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.image_path,
            ann_path=build_info.ann_path
        )

        return datasets


@registry.register_builder("sslidevqa")
class SingleSlideVQABuilder(DocumentVQABuilder):
    train_dataset_cls = SingleSlideVQADataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/doc/sslidevqa.yaml"}


@registry.register_builder("ocrvqa")
class OCRVQABuilder(DocumentVQABuilder):
    train_dataset_cls = OCRVQADataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/doc/ocrvqa.yaml"}