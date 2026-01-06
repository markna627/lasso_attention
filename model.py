import torch
import torch.nn as nn
class AttentionHead(nn.Module):
  ## Expect x = (d_k, seq_len)
  ## Expect d_model // num_heads == d_k (something nice)
  def __init__(self, d_model = 512, d_k = 64, attention = 'softmax'):
    super().__init__()
    self.attention = attention
    self.W_q = nn.Linear(d_model, d_k)
    self.W_k = nn.Linear(d_model, d_k)
    self.W_v = nn.Linear(d_model, d_k)
    self.attention_score = None
    self.cached_V = 0
  def forward(self, x, mask = None):
    Q = self.W_q(x)
    K = self.W_k(x)
    V = self.W_v(x)
    self.cached_V = V
    if self.attention == 'lasso':
      self.attention_score = lasso_score(Q, K, mask)
    elif self.attention == 'softmax':
      self.attention_score = softmax_score(Q, K, mask)
    return self.attention_score @ V


def softmax_score(Q, K, mask = None):
  qkt = Q @ K.permute(0, 2, 1)
  qkt = qkt/torch.sqrt(torch.tensor(K.shape[-1], dtype = qkt.dtype))
  if not(mask is None):
    mask = mask.to(qkt.device)
    qkt = qkt.masked_fill(mask, float('-inf'))
  softmaxed = torch.softmax(qkt, dim = 2)
  return softmaxed

def lasso_score(Q, K, mask = None):
  Q = Q/torch.sqrt(torch.tensor(K.shape[-1], dtype = K.dtype))
  K = K/torch.sqrt(torch.tensor(K.shape[-1], dtype = K.dtype))
  A = K.permute(0, 2, 1)
  alpha = torch.zeros(Q.shape[0], Q.shape[1], Q.shape[1], device = Q.device)
  for t in range(Q.shape[1]):
    A_t = A[:,:,:t+1] # causality - look at K.T up until time step t
    b_t = Q[:, t,:] # the query vector at test time t
    s = torch.linalg.matrix_norm(A_t, ord=2)
    eta = 0.9 / (s*s + 1e-8)
    alpha_t = ISTA(A_t, b_t, step_size = eta, sparsity = 5e-2, num_iter = 40)
    alpha[:, t, :t+1] = alpha_t #coefficients of keys for the query at t
  return alpha

def ISTA(Kt, Q, step_size, sparsity, num_iter, mask = None):
  alpha = torch.zeros(Kt.shape[0], Kt.shape[-1], device=Kt.device, dtype=Kt.dtype)
  for i in range(num_iter):
    '''
    gradient on smooth f
    '''
    Kt_alpha = Kt @ alpha.unsqueeze(-1)
    Kt_alpha = Kt_alpha.squeeze(-1)
    res = Kt_alpha-Q
    grad = Kt.transpose(-1,-2) @ res.unsqueeze(-1)
    grad = grad.squeeze(-1)
    step_size = step_size.view(-1, 1)
    z = alpha - step_size * grad

    '''
    soft-thresholding z (regularizer)
    '''
    alpha = soft_threshold(z, step_size * sparsity)
  return alpha

def soft_threshold(z, thres):
  return torch.sign(z) * torch.relu(torch.abs(z) - thres)

class FeedForward(nn.Module):
  def __init__(self, d_model, scale = 4):
    super().__init__()
    self.ff = nn.Sequential(
        nn.Linear(d_model, d_model * scale),
        nn.ReLU(),
        nn.Linear(d_model*scale, d_model)
    )
  def forward(self, x):
    out = self.ff(x)
    return out


class MultiHeadAttention(nn.Module):
  def __init__(self, d_model = 512, num_heads = 8, attention = 'softmax'):
    super().__init__()
    self.num_heads = num_heads
    self.attention = attention
    self.d_k = d_model//self.num_heads
    self.attn_heads = nn.ModuleList([AttentionHead(d_model, self.d_k, self.attention) for _ in range(self.num_heads)])
    self.fc = nn.Linear(d_model, d_model)
    self.cached_atten = 0
  def forward(self, x, mask = None):
    self.cached_atten = [attn_head(x, mask) for attn_head in self.attn_heads]
    out = self.cached_atten
    out = torch.cat(out, dim = -1)
    fc_out = self.fc(out)
    return fc_out

class Decoder(nn.Module):
  def __init__(self, d_model = 512, num_heads = 8, attention = 'softmax'):
    super().__init__()
    self.attention = attention
    self.masked_mha = MultiHeadAttention(d_model, num_heads, self.attention)
    self.layernorm1 = nn.LayerNorm(d_model)
    self.ff = FeedForward(d_model)
    self.layernorm2 = nn.LayerNorm(d_model)

  def forward(self, x, mask):
    masked_mha_out = self.masked_mha(x, mask = mask)
    res_masked_mha_out = x + masked_mha_out
    add_and_norm1 = self.layernorm1(res_masked_mha_out)
    ff_out = self.ff(add_and_norm1)
    res_ff_out = ff_out + add_and_norm1
    add_and_norm2 = self.layernorm2(res_ff_out)

    return add_and_norm2


class TransformerDecoder(nn.Module):
  def __init__(self, d_model = 512, num_heads = 8, num_layers = 6, attention = 'softmax'):
    super().__init__()
    self.num_layers = num_layers
    self.attention = attention
    self.decoders = nn.ModuleList([Decoder(d_model, num_heads, self.attention)for _ in range(self.num_layers)])
  def forward(self, x, mask):
    out = x
    for decoder in self.decoders:
      out = decoder(out, mask)
    return out


class PositionalEncoding(nn.Module):
  def __init__(self,d_model = 512, max_len = 1024):
    super().__init__()
    self.pos_num = torch.arange(0, max_len).unsqueeze(1).expand(max_len, d_model)
    self.pos_denom = 10000*torch.ones((max_len, d_model))
    self.exponent = 2*(torch.arange(0, d_model).unsqueeze(0).expand(max_len,d_model)//2)
    self.pos_denom = torch.pow(self.pos_denom, self.exponent/d_model)

    self.pe = self.pos_num/self.pos_denom
    self.res = torch.zeros((max_len, d_model))
    self.res[:, 0::2] = torch.sin(self.pe[:, 0::2])
    self.res[:, 1::2] = torch.cos(self.pe[:, 1::2])
  def forward(self,x):
    return x + self.res[:x.shape[1], :]



class Transformer(nn.Module):
  def __init__(self,
               vocab_size = 1024,
               d_model = 512,
               num_heads = 8,
               num_layers = 6,
               max_seq_len = 32,
               attention = 'softmax'):
    super().__init__()
    self.src_pe = PositionalEncoding(d_model, max_seq_len)
    self.src_embedding = nn.Embedding(vocab_size, d_model)
    self.decoder_block = TransformerDecoder(d_model, num_heads, num_layers, attention = attention)
    self.head = nn.Linear(d_model, vocab_size)

  def forward(self, src, mask):
    src_embed = self.src_embedding(src)
    src_pe_embed = self.src_pe(src_embed)
    decoder_out =self.decoder_block(src_pe_embed,mask)

    out = self.head(decoder_out)
    return out




