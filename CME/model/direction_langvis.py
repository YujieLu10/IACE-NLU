import torch
from icecream import ic
from torch._C import device
import torch.nn.functional as F

class DirectionalLanguageAndVision(torch.nn.Module):
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
        super(DirectionalLanguageAndVision, self).__init__()

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
        self.loss_fn = loss_fn
        self.dropout = torch.nn.Dropout(dropout_p)
        self.num_classes = num_classes
        self.type = task_type
        
    def forward(self, txt_pre, txt_hyp, img_pre, img_hyp, label=None):
        pre_text_features = torch.nn.functional.relu(txt_pre)
        hyp_text_features = torch.nn.functional.relu(txt_hyp)
        pre_image_features = torch.nn.functional.relu(img_pre)
        hyp_image_features = torch.nn.functional.relu(img_hyp)

        text_direction = (hyp_text_features - pre_text_features)
        text_direction_norm = text_direction.norm(dim=-1, keepdim=True)
        text_d = text_direction / text_direction_norm

        image_direction = (hyp_image_features - pre_image_features)
        image_direction_norm = image_direction.norm(dim=-1, keepdim=True)
        image_d = image_direction / image_direction_norm
        combined = torch.cat(
            [text_d, image_d], dim=1
        )
        fused = self.dropout(
            torch.nn.functional.relu(
            self.fusion(combined.float())
            )
        )
        dense_out = self.dense(fused)
        dense_out = self.dropout(dense_out)
        logits = self.fc(dense_out)
        pred = torch.nn.functional.tanh(logits)
        loss = (
            self.loss_fn(pred, label)
            if label is not None else label
        )
        return (logits, loss)