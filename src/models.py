import torch
import torch.nn as nn
import sentence_transformers 
from torch.utils.data.dataset import Dataset

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class BiEncoder(nn.Module):
    def __init__(self, args):
        super(BiEncoder, self).__init__()
        self.label_encoder = sentence_transformers.models.Transformer(args.model_name, max_seq_length=None)
        if args.share_weight:
            self.instance_encoder = self.label_encoder
        else:
            self.instance_encoder = sentence_transformers.models.Transformer(args.model_name, max_seq_length=None)
    
    def encode_doc(self, inst_tokens, inst_att_mask):
        inst_f = {'input_ids': inst_tokens, 'attention_mask': inst_att_mask}
        inst_emb = self.instance_encoder(inst_f)
        inst_emb = inst_emb['token_embeddings'] #[:, 0]
        inst_emb = mean_pooling(inst_emb, inst_att_mask)
        return inst_emb
        
    def encode_label(self,  label_tokens, label_att_mask):
        label_f = {'input_ids': label_tokens, 'attention_mask': label_att_mask}
        label_emb = self.label_encoder(label_f)
        label_emb = label_emb['token_embeddings']#[:, 0]
        label_emb = mean_pooling(label_emb, label_att_mask)
        return label_emb


    def forward(self,  inst_tokens, inst_att_mask, label_tokens, label_att_mask):
        inst_f = {'input_ids': inst_tokens, 'attention_mask': inst_att_mask}
        inst_emb = self.instance_encoder(inst_f)
        inst_emb = inst_emb['token_embeddings'] #[:, 0]
        inst_emb = mean_pooling(inst_emb, inst_att_mask)


        label_f = {'input_ids': label_tokens, 'attention_mask': label_att_mask}
        label_emb = self.label_encoder(label_f)
        label_emb = label_emb['token_embeddings'] #[:, 0]
        label_emb = mean_pooling(label_emb, label_att_mask)
        return inst_emb, label_emb
