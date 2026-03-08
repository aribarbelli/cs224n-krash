import torch

from einops import rearrange
from torch import nn
import math


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # LORA EXTENSION
    self.use_lora = config.use_lora
    self.lora_r = 8 # subject to change
    self.lora_alpha = 16 # subject to change
    # A should be random, as per LoRA paper. seems like default is random already
    self.query_lora_A = nn.Linear(config.hidden_size, self.lora_r, bias=False)
    self.value_lora_A = nn.Linear(config.hidden_size, self.lora_r, bias=False)
    # B should be initialized to 0, BA = 0
    self.query_lora_B = nn.Linear(self.lora_r, self.all_head_size, bias=False)
    nn.init.zeros_(self.query_lora_B.weight)
    self.value_lora_B = nn.Linear(self.lora_r, self.all_head_size, bias=False)
    nn.init.zeros_(self.value_lora_B.weight)

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj
  
  def transform_lora(self, x, linear_layer, A_layer, B_layer):
    # here, we do W0x + (alpha / r) * BAx instead of just W0x, as the lora paper suggests
    base_proj = linear_layer(x)
    lora_proj = B_layer(A_layer(x))
    scaling = (self.lora_alpha / self.lora_r)
    proj = base_proj + scaling * lora_proj
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    # (b, h, t, d_h) = (batch_size, num_attention_heads, sequence length, dimnesions/heads)
    b, h, t, d_h = key.shape

    # softmax((Q@K_t) / sqrt(dh)) @ v
    key_transposed = key.transpose(-1, -2)
    scores = query @ key_transposed
    scores = scores / math.sqrt(d_h)

    # causal mask: block attending to future tokens
    causal = torch.tril(torch.ones(t, t, device=scores.device)).bool()   # [t, t]
    causal = causal.view(1, 1, t, t)                                     # [1, 1, t, t]
    scores = scores.masked_fill(~causal, float("-inf"))
    # padding mask (additive): attention_mask is [b, 1, 1, t] with 0 for real tokens and -10000 for pads
    scores = scores + attention_mask

    out = torch.nn.functional.softmax(scores, dim=-1) 
    out = self.dropout(out) @ value 
    # out has shape (b, h, t, d/h)
    out = out.transpose(1, 2) # (b, t, h, d/h)
    hidden_size = h * d_h # hidden size 
    out = out.reshape(b, t, hidden_size) # (b, t, d)

    return out


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    if self.use_lora:
        # lora paper only adapts Wq and Wv... doing just that for now
        query_layer = self.transform_lora(hidden_states, self.query,
                                    self.query_lora_A, self.query_lora_B)
        value_layer = self.transform_lora(hidden_states, self.value,
                                    self.value_lora_A, self.value_lora_B)
    else:
        query_layer = self.transform(hidden_states, self.query)
        value_layer = self.transform(hidden_states, self.value)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
