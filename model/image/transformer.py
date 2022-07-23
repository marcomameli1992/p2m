from transformers import ViTFeatureExtractor, ViTModel
import torch
from torch import nn

class Extractor(nn.Module):
    def __init__(self, transformer_name) -> None:
        super().__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(transformer_name)
        # google/vit-base-patch16-224-in21k, google/vit-base-patch16-224, google/vit-base-patch16-384, google/vit-base-patch32-384, google/vit-base-patch32-224-in21k, google/vit-large-patch16-224-in21k, google/vit-large-patch16-224, google/vit-large-patch32-224-in21k, google/vit-large-patch32-384, google/vit-large-patch16-384, google/vit-large-patch16-384, google/vit-huge-patch14-224-in21k
        # facebook/deit-small-patch16-224, facebook/deit-tiny-patch16-224, facebook/deit-base-patch16-224, facebook/deit-tiny-distilled-patch16-224, facebook/deit-base-patch16-384
        self.model = ViTModel.from_pretrained(transformer_name)
        
    def forward(self, x):
        inputs = self.feature_extractor(x[0], return_tensors="pt")
        inputs.data['pixel_values'] = inputs.data['pixel_values'].to('cuda:0')
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, output_attentions=True, interpolate_pos_encoding=True)
        return outputs.attentions[0], outputs.attentions[int(len(outputs.attentions)/2)], outputs.attentions[-1]