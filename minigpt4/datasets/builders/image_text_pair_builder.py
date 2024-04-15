import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from minigpt4.datasets.datasets.laion_dataset import LaionDataset
from minigpt4.datasets.datasets.cc_sbu_dataset import CCSBUDataset, CCSBUAlignDataset
from minigpt4.datasets.datasets.vg_dataset import ReferVisualGenomeDataset
from minigpt4.datasets.datasets.open_images import OpenImageDataset,OpenBboxToObjectDataset
from minigpt4.datasets.datasets.locna_dataset import LocNaCOCODataset
from minigpt4.datasets.datasets.llava_dataset import LlavaDetailDataset, LlavaReasonDataset, LlavaConversationDataset
from minigpt4.datasets.datasets.lvis_dataset import LVISBBOXDataset,LVISBboxToObjectDataset
from minigpt4.datasets.datasets.text_caps import TextCapBboxToObjectDataset, TextCapDataset
from minigpt4.datasets.datasets.coco_caption import COCOCapDataset,COCOCapEvalDataset
from minigpt4.datasets.datasets.coyo_dataset import COYOCaptionWDSDataset,COYOBoxToPhraseWDSDataset,COYOPhraseToBoxWDSDataset
# , COYOBBoxPhraseDataset
from minigpt4.datasets.datasets.grounded_detailed_image_caption_dataset import GroundedDetailDataset
from minigpt4.datasets.datasets.reasoning_dataset import ReasoningDataset
from minigpt4.datasets.datasets.video_datasets import CMDVideoDataset, WebVidDataset,VideoChatGPTDataset
from minigpt4.datasets.datasets.cot import CoTDataset
from minigpt4.datasets.datasets.unnatural_instruction import UnnaturalDataset
from minigpt4.datasets.datasets.caption_reasoning import CaptionReasonDataset
from minigpt4.datasets.datasets.aok_vqa_reasoning_datasets import AOKVQAReasoningDataset
from minigpt4.datasets.datasets.paint_dataset import PaintPTCOCODataset, PaintRLCOCODataset, PaintPixelCOCODataset, SegReferCOCODataset, PaintLanRLOpaqueCOCODataset
from minigpt4.datasets.datasets.nav_dataset import NavR2RDataset

@registry.register_builder("yifan_reasoning")
class LlavaDetailBuilder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQAReasoningDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/aokvqa_reasoning/defaults.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=build_info.ann_path,
            vis_root=build_info.image_path,
        )

        return datasets


@registry.register_builder("caption_reasoning")
class CaptionReasoningBuilder(BaseDatasetBuilder):
    train_dataset_cls = CaptionReasonDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/mm_reasoning/mm_reasoning.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls

        # print("ann_path",build_info.ann_path)
        # print("vis root",build_info.image_path )

        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors['train'],
            text_processor=self.text_processors['train'],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )


        return datasets


@registry.register_builder("unnatural_instruction")
class UnnaturalInstructionBuilder(BaseDatasetBuilder):
    train_dataset_cls = UnnaturalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nlp/unnatural_instruction.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_path=build_info.ann_path,
        )

        return datasets

@registry.register_builder("cot")
class CoTBuilder(BaseDatasetBuilder):
    train_dataset_cls = CoTDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nlp/cot.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_path=build_info.ann_path,
        )

        return datasets




@registry.register_builder("coco_caption")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset
    eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/caption.yaml",
        "eval": "configs/datasets/coco/caption.yaml",
    }


@registry.register_builder("open_images")
class OpenImageBuilder(BaseDatasetBuilder):
    train_dataset_cls = OpenImageDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/open_images/default.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets



@registry.register_builder("open_images_bbox_to_object")
class OpenBboxToObjectuilder(BaseDatasetBuilder):
    train_dataset_cls = OpenBboxToObjectDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/open_images/default_bbox.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("lvis_images_bbox")
class LVISBBOxBuilder(BaseDatasetBuilder):
    train_dataset_cls = LVISBBOXDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/lvis/default_bbox.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets



@registry.register_builder("lvis_bbox_to_object")
class LVISBBoxToObjectBuilder(BaseDatasetBuilder):
    train_dataset_cls = LVISBboxToObjectDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/lvis/bbox_to_object.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets




@registry.register_builder("spatial_reasoning")
class ReasoningBuilder(BaseDatasetBuilder):
    train_dataset_cls = ReasoningDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/reasoning/default.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )

        return datasets





@registry.register_builder("textcaps_caption")
class TextcapCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = TextCapDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/textcaps/caption.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )

        return datasets





@registry.register_builder("coyo_caption")
class CoyoCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = COYOCaptionWDSDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/coyo/default.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets



@registry.register_builder("coyo_bbox_phrase")
class CoyoBboxPhraseBuilder(BaseDatasetBuilder):
    train_dataset_cls = COYOBoxToPhraseWDSDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/coyo/bbox_phrase.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("coyo_phrase_bbox")
class CoyoBboxPhraseBuilder(BaseDatasetBuilder):
    train_dataset_cls = COYOPhraseToBoxWDSDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/coyo/phrase_bbox.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("cc_sbu_align")
class CCSBUAlignBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUAlignDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cc_sbu/align.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'filter_cap.json')],
            vis_root=os.path.join(storage_path, 'image'),
        )

        return datasets

@registry.register_builder("cc_sbu")
class CCSBUBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/cc_sbu/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("textcaps_ocr")
class TextcapCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = TextCapBboxToObjectDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/textcaps/ocr.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
            )

        return datasets





@registry.register_builder("laion")
class LaionBuilder(BaseDatasetBuilder):
    train_dataset_cls = LaionDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/laion/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("locna_coco")
class LocNaCOCOBuilder(BaseDatasetBuilder):
    train_dataset_cls = LocNaCOCODataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_locna.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        ann_paths = build_info.annotations.train.storage

        datasets = dict()

        for ann_path in ann_paths:
            if not os.path.exists(ann_path):
                warnings.warn("storage path {} does not exist.".format(ann_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=ann_paths,
            vis_root=build_info.images.storage,
        )

        return datasets


@registry.register_builder("llava_detail")
class LlavaDetailBuilder(BaseDatasetBuilder):
    train_dataset_cls = LlavaDetailDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/llava/detail.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )

        return datasets

@registry.register_builder("grounded_detailed_image_caption")
class GroundedCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = GroundedDetailDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/grounded_image_caption/default.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )

        return datasets




@registry.register_builder("llava_reason")
class LlavaReasonBuilder(BaseDatasetBuilder):
    train_dataset_cls = LlavaReasonDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/llava/reason.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )

        return datasets





@registry.register_builder("llava_conversation")
class LlavaReasonBuilder(BaseDatasetBuilder):
    train_dataset_cls = LlavaConversationDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/llava/conversation.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )

        return datasets


class AllRefCOCOBuilder(BaseDatasetBuilder):

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        image_path = build_info.image_path
        ann_path = build_info.ann_path

        datasets = dict()

        if not os.path.exists(image_path):
            warnings.warn("image path {} does not exist.".format(image_path))
        if not os.path.exists(ann_path):
            warnings.warn("ann path {} does not exist.".format(ann_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_path=ann_path,
            vis_root=image_path,
            dataset=build_info.dataset,
            splitBy=build_info.splitBy
        )

        return datasets


@registry.register_builder("refvg")
class RefVisualGenomeBuilder(BaseDatasetBuilder):
    train_dataset_cls = ReferVisualGenomeDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vg/ref.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        data_dir = build_info.data_dir
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            data_dir=data_dir,
        )

        return datasets


@registry.register_builder("cmd_video")
class CMDVideoBuilder(BaseDatasetBuilder):
    train_dataset_cls = CMDVideoDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cmd_video/default.yaml",
    }

    def build_datasets(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        self.build_processors()

        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            vis_root=build_info.vis_root,
            ann_paths=build_info.ann_paths,
            cc_path=build_info.cc_path,
            model_name= 'llama2',
        )

        return datasets


@registry.register_builder("webvid")
class WebVidBuilder(BaseDatasetBuilder):
    train_dataset_cls = WebVidDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/webvid/default.yaml",
    }

    def build_datasets(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        self.build_processors()

        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            vis_root=build_info.vis_root,
            ann_paths=build_info.ann_paths,
            subtitles_path=build_info.subtitles_path,
            model_name= 'llama2',
        )

        return datasets


@registry.register_builder("video_chatgpt")
class VideoChatGPTBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoChatGPTDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/video_chatgpt/default.yaml",
    }
    print(DATASET_CONFIG_DICT)

    def build_datasets(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        self.build_processors()

        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            vis_root=build_info.vis_root,
            ann_paths=build_info.ann_paths,
            subtitles_path=build_info.subtitles_path,
            model_name='llama2'
        )

        return datasets
    
@registry.register_builder("Name of the builder as in the config file")
class VideoTemplateBuilder(BaseDatasetBuilder):
    train_dataset_cls = ... # Add the dataset class here

    DATASET_CONFIG_DICT = {
        "default": "path to the config file",
    }
    print(DATASET_CONFIG_DICT)

    def build_datasets(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        self.build_processors()

        build_info = self.config.build_info # information from the config file
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"], # Add the vis_processor here
            text_processor=self.text_processors["train"], # Add the text_processor here
            vis_root=build_info.vis_root, # Add videos path here
            ann_paths=build_info.ann_paths, # Add annotations path here
            subtitles_path=build_info.subtitles_path, # Add subtitles path here
            model_name='llama2' # Add model name here (llama2 or mistral)
        )

        return datasets

@registry.register_builder("r2r")
class NavR2RBuilder(BaseDatasetBuilder):
    train_dataset_cls = NavR2RDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nav/r2r.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            data_root=build_info.data_root
        )

        return datasets


@registry.register_builder("paintcoco")
class PaintPTCOCOBuilder(BaseDatasetBuilder):
    train_dataset_cls = PaintPTCOCODataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/paint/coco.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        img_root = build_info.img_root
        stroke_root = build_info.stroke_root
        max_step = build_info.max_step

        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            img_root=img_root,
            stroke_root=stroke_root,
            max_step=max_step
        )

        return datasets


class PaintRLCOCOBuilderBase(BaseDatasetBuilder):
    train_dataset_cls = PaintRLCOCODataset

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        img_root = build_info.img_root
        stroke_root = build_info.stroke_root
        max_step = build_info.max_step
        single_stroke = build_info.single_stroke

        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            img_root=img_root,
            stroke_root=stroke_root,
            max_step=max_step,
            single_stroke=single_stroke
        )

        return datasets


@registry.register_builder("paintrlcoco")
class PaintRLCOCOBuilder(PaintRLCOCOBuilderBase):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/paint/rl_coco.yaml",
    }


@registry.register_builder("paintrlscoco")
class PaintRLSCOCOBuilder(PaintRLCOCOBuilderBase):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/paint/rls_coco.yaml",
    }


@registry.register_builder("paintlanrlsococo")
class PaintLanRLOpaqueCOCOBuilder(BaseDatasetBuilder):
    train_dataset_cls = PaintLanRLOpaqueCOCODataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/paint/lan_rls_o_coco.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        img_root = build_info.img_root
        stroke_root = build_info.stroke_root
        max_step = build_info.max_step
        single_stroke = build_info.single_stroke
        ann_path = build_info.ann_path

        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            img_root=img_root,
            stroke_root=stroke_root,
            ann_path=ann_path,
            max_step=max_step,
            single_stroke=single_stroke
        )

        return datasets


class PaintPixelCOCOBuilder(BaseDatasetBuilder):
    train_dataset_cls = PaintPixelCOCODataset

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)
        res = build_info.res

        datasets = dict()
        split = 'train'

        # annotation path
        ann_paths = ann_info.get(split).storage
        if isinstance(ann_paths, str):
            ann_paths = [ann_paths]

        # visual data storage path
        vis_path = os.path.join(vis_info.storage, split)

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=ann_paths,
            vis_root=vis_path,
            res=res
        )

        return datasets


@registry.register_builder("paintpixelcoco32")
class PaintPixelCOCO32Builder(PaintPixelCOCOBuilder):
    train_dataset_cls = PaintPixelCOCODataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/paint/pixel_coco_32.yaml",
    }


@registry.register_builder("paintpixelcoco64")
class PaintPixelCOCO64Builder(PaintPixelCOCOBuilder):
    train_dataset_cls = PaintPixelCOCODataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/paint/pixel_coco_64.yaml",
    }


class AllSegRefCOCOBuilder(BaseDatasetBuilder):
    train_dataset_cls = SegReferCOCODataset

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        image_path = build_info.image_path
        ann_path = build_info.ann_path
        res = build_info.res

        datasets = dict()

        if not os.path.exists(image_path):
            warnings.warn("image path {} does not exist.".format(image_path))
        if not os.path.exists(ann_path):
            warnings.warn("ann path {} does not exist.".format(ann_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_path=ann_path,
            vis_root=image_path,
            res=res,
            dataset=build_info.dataset,
            splitBy=build_info.splitBy
        )

        return datasets


@registry.register_builder("segrefcoco32")
class SegRefCOCO32Builder(AllSegRefCOCOBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/paint/segrefcoco32.yaml",
    }


@registry.register_builder("segrefcocop32")
class SegRefCOCOP32Builder(AllSegRefCOCOBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/paint/segrefcocop32.yaml",
    }


@registry.register_builder("segrefcocog32")
class SegRefCOCOG32Builder(AllSegRefCOCOBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/paint/segrefcocog32.yaml",
    }


@registry.register_builder("segrefcoco64")
class SegRefCOCO64Builder(AllSegRefCOCOBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/paint/segrefcoco64.yaml",
    }


@registry.register_builder("segrefcocop64")
class SegRefCOCOP64Builder(AllSegRefCOCOBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/paint/segrefcocop64.yaml",
    }


@registry.register_builder("segrefcocog64")
class SegRefCOCOG64Builder(AllSegRefCOCOBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/paint/segrefcocog64.yaml",
    }
