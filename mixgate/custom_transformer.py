# 新建custom_transformer.py文件
import torch
import torch.nn as nn

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights

class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        
    def forward(self, src, mask=None, src_key_padding_mask=None):
        attn_weights = []
        for layer in self.layers:
            src, attn = layer(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attn_weights.append(attn)
        return src, attn_weights