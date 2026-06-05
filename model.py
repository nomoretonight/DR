import torch
import torch.nn as nn
import timm

class DRMultiHeadModel(nn.Module):
    def __init__(self, backbone_name='efficientnet_b3', pretrained=True, num_classes=5, dropout=0.5):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=False, num_classes=0)
        in_features = self.backbone.num_features
        self.dropout = nn.Dropout(p=dropout)
        self.fc_multiclass = nn.Linear(in_features, num_classes)   # 0-4
        self.fc_binary = nn.Linear(in_features, 1)                 # referable
        # init
        nn.init.normal_(self.fc_multiclass.weight, 0, 0.01)
        nn.init.normal_(self.fc_binary.weight, 0, 0.01)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        out_multiclass = self.fc_multiclass(feat)
        out_binary = self.fc_binary(feat).squeeze(1)
        return out_multiclass, out_binary   # multiclass logits, referable logits
