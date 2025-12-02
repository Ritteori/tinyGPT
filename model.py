import torch
from torch import nn
import math

class CausalMSA(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, max_seq_len=128, attn_dropout=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, 'Embed_dim / num_heads must be integer'
        
        self.qkv = nn.Linear(embed_dim,embed_dim * 3)

        mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)  # moves with .to(device)
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module,nn.Linear):
                nn.init.xavier_normal_(module.weight)
                
                if module.bias is not None:   
                    nn.init.zeros_(module.bias)
    
    def forward(self,x,past_kv = None):
        
        # Input: (bs,sequence_length,embed_dim)
        
        bs, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim, f'Expected {self.embed_dim}, got {embed_dim}'
        
        x = self.qkv(x) # (bs,sequence_length,3 * embed_dim)
        x = x.view(bs,seq_len,3,self.num_heads,self.head_dim) # (bs,sequence_length,3,num_heads,head_dim)
        x = x.permute(2,0,3,1,4).contiguous() # (3,bs,num_heads,sequence_length,head_dim)
        
        Q = x[0] # (bs,num_heads,sequence_length,head_dim)
        K = x[1] # (bs,num_heads,sequence_length,head_dim)
        V = x[2] # (bs,num_heads,sequence_length,head_dim)
        
        if past_kv is not None:
            past_K, past_V = past_kv
            K = torch.cat([past_K, K], dim=2)
            V = torch.cat([past_V, V], dim=2)
        
        total_len = K.size(2)
        past_len = total_len - seq_len
        
        scores = Q @ K.transpose(-1,-2) / self.head_dim ** 0.5 # (bs,num_heads,sequence_length,sequence_length)
        
        mask = self.causal_mask[past_len: past_len + seq_len, :total_len]  # (seq_len, total_len)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,seq_len,total_len) -> will broadcast over bs & heads
        scores = scores.masked_fill(mask, float('-inf'))
    
        probs = torch.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)
        
        out = probs @ V # (bs,num_heads,sequence_length,head_dim)
        out = out.permute(0,2,1,3).contiguous() # (bs,sequence_length,num_heads,head_dim)
        out = out.view(bs,seq_len,embed_dim) # (bs,sequence_length,embed_dim)
        
        return out, (K,V)
    
class FeedForward(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=4, dropout=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim,embed_dim*hidden_dim),
            nn.GELU(),
            nn.Linear(embed_dim*hidden_dim,embed_dim),
            nn.Dropout(dropout)
        )

        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module,nn.Linear):
                nn.init.xavier_normal_(module.weight)
                
                if module.bias is not None:   
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module,nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self,x):
        
        out = self.ff(x)
        
        return out
        
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, hidden_dim=4, dropout=0.2, max_seq_len=128, attn_dropout=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.first_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.attention = CausalMSA(embed_dim,num_heads,max_seq_len,attn_dropout)
        self.second_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.ff = FeedForward(embed_dim,hidden_dim,dropout)

        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module,nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
  
    def forward(self, x, past_kv=None):
        
        x_norm1 = self.first_norm(x)
        attn_out, kv = self.attention(x_norm1, past_kv)
        x = x + attn_out
        
        x_norm2 = self.second_norm(x)
        ff_out = self.ff(x_norm2)
        x = x + ff_out
        
        return x, kv
    
class TinyGpt(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, hidden_dim=4, dropout=0.2, depth=6, max_seq_len=128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.token_emb = nn.Embedding(vocab_size,embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len,embed_dim)
        self.vocab_size = vocab_size
        
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(TransformerBlock(embed_dim,num_heads,hidden_dim,dropout,max_seq_len))
        
        self.final_norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.final_proj = nn.Linear(embed_dim,vocab_size,bias=False)
        
        self.max_seq_len = max_seq_len

        self._init_weights()
        
    def _init_weights(self):
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'ff' in name and '2' in name:  # Второй слой FFN
                    nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * len(self.layers)))
                elif 'qkv' in name or 'final_proj' in name:
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(self,idx, past_kvs=None, return_past=False):
        
        bs, seq_len = idx.size()
        assert seq_len <= self.max_seq_len, 'Seq too long'
        
        past_len = 0
        if past_kvs is not None and past_kvs[0] is not None:
            past_len = past_kvs[0][0].size(2)
        
        pos = torch.arange(past_len, past_len + seq_len, device=idx.device).unsqueeze(0)# (1, seq_len)
        
        tok_emb = self.token_emb(idx) # (bs, seq_len, embed_dim)
        tok_pos = self.pos_emb(pos) # (1, seq_len, embed_dim)
        x = tok_emb + tok_pos # (bs, seq_len, embed_dim)
        
        if past_kvs is None:
            past_kvs = [None] * len(self.layers)

        new_past_kvs = []
        for layer, past in zip(self.layers, past_kvs):
            x, kv = layer(x, past)
            new_past_kvs.append(kv)

        x = self.final_norm(x)
        logits = self.final_proj(x)  # (bs, seq_len, vocab_size)

        if return_past:
            return logits, new_past_kvs
        else:
            return logits