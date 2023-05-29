"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # sparsity parameters
        self.l1_lambda = config.lmbda
        self.alphas_from = config.alphas_from # either 'x' or 'q'
        self.n_dims_per_head = int(config.n_embd // config.n_head)
        self.use_sigmoid = config.use_sigmoid
        self.alphas_mlp = nn.Sequential(nn.Linear(config.n_embd, config.n_alpha_mlp), nn.ReLU(), nn.Linear(config.n_alpha_mlp, config.n_head))
        if self.l1_lambda > 0.0: 
            self.alphas_mlp[2].bias.data.fill_(1.0) # init bias to high value so that at first the "gates" are open
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.sequence_length, config.sequence_length))
                                        .view(1, 1, config.sequence_length, config.sequence_length))

    def forward(self, x, alpha_th=None, drop_k=None, get_alphas=False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        # compute alphas
        num_head_pruned, alphas = (0, 0), None
        if self.l1_lambda > 0 or drop_k is not None:
            q_ = q if self.alphas_from == 'q' else x
            if self.use_sigmoid:
                alphas = torch.sigmoid(self.alphas_mlp(q_))
            else:
                alphas = torch.max(torch.min(self.alphas_mlp(q_).abs(), torch.ones(1, device=x.device)), torch.zeros(1, device=x.device)) 
            l1_penalty = alphas.view(B,-1).abs().sum(dim=-1) # l1 loss computed for each sequence 
            if alpha_th is not None: # then we set the alphas under alpha_th to 0
                mask = alphas <= alpha_th
                num_head_pruned = (mask.sum().float().item(), float(mask.numel()))
                alphas[mask] = 0.0
            elif drop_k is not None:
                if self.l1_lambda == 0 :
                    alphas = torch.rand_like(alphas)
                alphas = alphas.view(B,-1)
                values, _ = torch.sort(alphas) # sorting the alphas for all heads
                th = values[:, drop_k].view(B,1) # for each sequence in the batch, finding the threshold value s.t. drop_k alphas are lower
                alphas[alphas < th] = 0.0 # set drop_k alphas to 0 for each sequence
                if self.l1_lambda == 0 :
                    alphas[alphas >= th] = 1.0
                alphas = alphas.view(B, T, self.n_head)
            alphas = alphas.unsqueeze(-1).expand(B, T, self.n_head, self.n_dims_per_head)
            alphas = alphas.reshape(*x.shape)
        else: 
            l1_penalty = torch.zeros(1, device=x.device)
        # reshape query, key, values
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        if self.l1_lambda > 0 or drop_k is not None:
            y = self.resid_dropout(self.c_proj(alphas * y)) # multiply with the alpha values
        else: 
            y = self.resid_dropout(self.c_proj(y))
        alphas = alphas if get_alphas else None
        return y, l1_penalty, num_head_pruned, alphas


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, alpha_th=None, drop_k=None, get_alphas=False):
        att, l1_penalty, num_head_pruned, alphas = self.attn(self.ln_1(x), alpha_th=alpha_th, drop_k=drop_k, get_alphas=get_alphas)
        x = x + att
        x = x + self.mlp(self.ln_2(x))
        return x, l1_penalty, num_head_pruned, alphas
    

class GPTSparseHeadsQ(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")
        # sparsity parameters
        self.l1_lambda = config.lmbda

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.sequence_length, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, alpha_th=None, drop_k=None, get_logits=False, get_alphas=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.sequence_length, f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        l1_penalties, alphas_per_layer, num_head_pruned_per_layer, num_heads_per_layer = [], [], [], []
        for block in self.transformer.h:
            x, l1_penalty, (nhp, nh), alphas = block(x, alpha_th=alpha_th, drop_k=drop_k, get_alphas=get_alphas)
            l1_penalties.append(l1_penalty)
            #alphas_per_layer.append(alphas)
            num_head_pruned_per_layer.append(nhp)
            num_heads_per_layer.append(nh)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            l1_loss = torch.stack(l1_penalties).sum(dim=0).mean()
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            loss = ce_loss + self.l1_lambda * l1_loss
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss, ce_loss, l1_loss = None, torch.zeros(1), torch.zeros(1)

        logits = logits if get_logits else None
        alphas_per_layer = alphas_per_layer if get_alphas else None

        return {'logits': logits, 'loss': loss, 'ce_loss': ce_loss.cpu().item(), 'l1_loss': l1_loss.cpu().item(), 
                'alphas': alphas_per_layer, 'num_head_pruned_per_layer': num_head_pruned_per_layer, 
                'num_heads_per_layer': num_heads_per_layer}

    def crop_sequence_length(self, sequence_length):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert sequence_length <= self.config.sequence_length
        self.config.sequence_length = sequence_length
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:sequence_length])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:sequence_length,:sequence_length]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # TODO
        pass

    def get_parameter_group_specs(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        return [
            {"params": sorted(list(decay))},
            {"params": sorted(list(no_decay)), "weight_decay": 0.0},
        ]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, alpha_th=None, drop_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at sequence_length
            idx_cond = idx if idx.size(1) <= self.config.sequence_length else idx[:, -self.config.sequence_length:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond, alpha_th=alpha_th, drop_k=drop_k, get_logits=True)['logits']
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    @torch.no_grad()
    def generate_from_string(self, in_str, max_new_tokens, temperature=1.0, top_k=None, alpha_th=None, drop_k=None):
        idx = torch.tensor(self.tokenizer.encode(in_str, allowed_special={"<|endoftext|>"})).view(1,-1).to(self.lm_head.weight.device)
        out_idx = self.generate(idx, max_new_tokens, temperature, top_k, alpha_th=alpha_th, drop_k=drop_k).view(-1).to('cpu').numpy()
        return self.tokenizer.decode(out_idx)
