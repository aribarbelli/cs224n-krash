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

  def attention(self, key, query, value, attention_mask):

    #IMPLEMENT THIS FUNCTION!!!!!

    print("k shape is:", key.shape)
    print("q shape is:", query.shape)
    print("v shape is:", value.shape)
    # b = batch size
    # h = num_attention_heads
    # t = sequence length
    # d_head = attention_head_size = dimensions / num_heads

    # (b, h, t, d_h) = (batch_size, num_attention_heads, sequence length, dimnesions/heads)
    b, h, t, d_h = key.shape
    print(key.shape)
    print(attention_mask.shape)

    # softmax((Q@K_t) / sqrt(dh)) @ v
    key_transposed = key.transpose(-1, -2)
    scores = query @ key_transposed
    scores = scores / math.sqrt(d_h)

    # causal masking
    mask = attention_mask[:, :, :t, :t] # slicing to make sure that the shape is (1, 1, context_len, context_len)
    scores = scores.masked_fill(mask == 0, float("-inf")) # need to use large negative number for softmax

    out = torch.nn.functional.softmax(scores, dim=-1) @ value
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
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
