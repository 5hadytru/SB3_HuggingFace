from transformers import ViTFeatureExtractor, ViTModel
from torch import nn
from torch.cuda.amp import autocast


class ViTWrapper(nn.Module):
    def __init__(self, pretrained_model_name):
        super(ViTWrapper, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(pretrained_model_name)
        self.vit_model = ViTModel.from_pretrained(pretrained_model_name)
        
    def forward(self, images):
        pixel_values = self.feature_extractor(images, return_tensors="pt")["pixel_values"]
        with autocast():
            outputs = self.vit_model(pixel_values)
        return outputs.last_hidden_state[:, 0]
