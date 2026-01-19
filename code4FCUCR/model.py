import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class Dice(nn.Module):
    def __init__(self, input_dim, eps=1e-9):
        super(Dice, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(input_dim)) 
        self.eps = eps
        self.bn = nn.BatchNorm1d(input_dim, eps=eps)

    def forward(self, x):
       
        if x.dim() == 2:
            x_normed = self.bn(x)
        else:
    
            orig_shape = x.shape
            x = x.contiguous().view(-1, x.shape[-1])
            x_normed = self.bn(x)
            x_normed = x_normed.view(orig_shape)

        p = torch.sigmoid(x_normed)
        return p * x + (1 - p) * self.alpha * x
class LinearLayer(nn.Module):

    def __init__(self, input_dimension, num_classes, bias=True):
        super(LinearLayer, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.fc = nn.Linear(input_dimension, num_classes, bias=bias)

    def forward(self, x):
        return self.fc(x)


class MLPForLogFeats(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPForLogFeats, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)  

        self.dice = Dice(hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
       
        batch_size, seq_len, input_dim = x.size()
        x = x.view(-1, input_dim)  
        x = self.fc1(x) 
        x = self.dice(x)
        x = self.fc2(x) 
      
        x = x.view(batch_size, seq_len, -1)  
        return x.squeeze(-1)


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs  
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, session_num, item_num, args):
        super(SASRec, self).__init__()

        self.session_num = session_num
        self.item_num = item_num

        self.dev = args.device


        
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()


        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.mlp4logfeats = MLPForLogFeats(input_dim=2*args.hidden_units, hidden_dim=80, output_dim=50)

        for _ in range(args.num_blocks):    

            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)


    def log2feats(self, log_seqs): # TODO: fp64 and int64 as default in python, trim?

        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= (log_seqs != 0)     

        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] 
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)
                                            
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats


    def forward(self, user_ids, log_seqs, KB_seqs, pos_seqs, neg_seqs): # for training

        log_feats = self.log2feats(log_seqs)

        assert neg_seqs.any() < self.item_emb.num_embeddings, "索引超出范围！"

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        B, L, D = log_feats.shape  

        stacked = torch.stack([log_feats, KB_seqs], dim=2)
        KB_con = stacked.view(B, L, D * 2)



        KB_con_new = self.mlp4logfeats(KB_con)
      

        pos_logits = (KB_con_new * pos_embs).sum(dim=-1)
        neg_logits = (KB_con_new * neg_embs).sum(dim=-1)



        return pos_logits, neg_logits, log_feats  


    def predict(self, user_ids, log_seqs, KB_seqs, item_indices): 

        log_feats = self.log2feats(log_seqs) 

        B, L, D = log_feats.shape  

        stacked = torch.stack([log_feats, KB_seqs], dim=2)

        KB_con = stacked.view(B, L, D * 2)
        KB_con_new = self.mlp4logfeats(KB_con)
        final_feat = KB_con_new[:, -1, :] 

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) 
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits # preds # (U, I)








