import torch
from icecream import ic
from torch._C import device
import torch.nn.functional as F
import numpy as np

class LanguageAndVisionConcat(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        loss_fn,
        language_feature_dim,
        vision_feature_dim,
        fusion_output_size,
        dropout_p,
        task_type="classification",
    ):
        super(LanguageAndVisionConcat, self).__init__()
        self.fusion = torch.nn.Linear(
            in_features=(language_feature_dim + vision_feature_dim), 
            out_features=fusion_output_size
        )
        self.dense = torch.nn.Linear(
            in_features=fusion_output_size, 
            out_features=fusion_output_size
        )
        self.fc = torch.nn.Linear(
            in_features=fusion_output_size, 
            out_features=num_classes
        )
        self.num_classes = num_classes
        self.type = task_type
        self.loss_fn = loss_fn
        self.dropout = torch.nn.Dropout(dropout_p)
        
    def forward(self, image, text=None, label=None):
        image_features = torch.nn.functional.relu(
            image
        )
        if text is not None:
            text_features = torch.nn.functional.relu(
                text
            )
            combined = torch.cat(
                [image_features, text_features], dim=1
            )
            fused = self.dropout(
                torch.nn.functional.relu(
                self.fusion(combined.float())
                )
            )
        else:
            fused = self.dropout(
                image_features.float()
            )
        dense_out = self.dense(fused)
        dense_out = self.dropout(dense_out)
        logits = self.fc(dense_out)
        logits = torch.nn.functional.tanh(logits)
        loss = (
            self.loss_fn(logits, label)
            if label is not None else label
        )
        return (logits, loss)