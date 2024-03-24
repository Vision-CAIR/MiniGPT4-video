import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionEncoder(nn.Module):
    def __init__(self, encoder_name="openai/clip-vit-large-patch14",  delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_encoder_name = encoder_name
        # self.select_layer = args.mm_vision_select_layer
        # self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.select_layer = -1
        self.select_feature = "patch"
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_encoder_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_encoder_name)
        self.vision_encoder = CLIPVisionModel.from_pretrained(self.vision_encoder_name)
        self.vision_encoder.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, :]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_encoder(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_encoder(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            # print("image feature shape", image_features.shape)
            # print(type(image_forward_outs))
            # print(type(image_forward_outs.shape))
            # image_features = image_forward_outs.to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_encoder.dtype

    @property
    def device(self):
        return self.vision_encoder.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_encoder.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2