#!/usr/bin/env python3
"""
Truly multilingual parametric eMG -- v2, action-head removed.

WHY v2.  The v1 core ran TWO overlapping structural mechanisms at once: (a) the
continuous Merge/Move gates feeding a universal composer, and (b) a discrete
attach/push/pop action head (RNNG-style; Dyer et al. 2016) mixing three composer
branches via a Gumbel-softmax.  A health probe on a 300M v1 checkpoint showed the
action head COLLAPSED (attach=0.00, push=0.99, pop=0.01; entropy 0) while the
switchboard->gate coupling stayed strong (||param_to_gate||=4.13 vs
||param_to_action||=0.21).  The discrete head is a high-variance bottleneck on top
of an already-expressive continuous path, so the optimiser does the work in the
gates and pins the actions to the single most stable branch ("push", whose output
ignores m and the cache).  Net effect: the actions are decorative AND, because
"push" reads `fresh` rather than the cache, the Move (displacement / re-access)
operation -- the eMG-distinctive one, the substrate for filler-gap and binding --
was effectively off (pop ~ 0).

v2 keeps the eMG idea and drops the redundant RNNG head:
  * controller emits Merge and Move gates only (no action_head, no param_to_action);
  * ONE continuous update per step: Merge the working state with (gated input + moved
    material), with the cache written/consumed continuously -- Move is now ALWAYS
    partially active and can no longer be switched off by a collapsing selector;
  * switchboard (InputInferredParameters) is UNCHANGED, so probe_parameters_emg.py and
    any LiP supervision attach exactly as before -- the live
    switchboard->gate pathway is the one we keep.

This changes the parameter shapes (action_head / param_to_action are gone), so v1
checkpoints do NOT load into v2; this needs a fresh run.  model_type is bumped to
"emg_ml" and auto_map points here.

  "model_type": "emg_ml",
  "architectures": ["EMGForCausalLM"],
  "auto_map": {"AutoConfig": "emg_multilingual.EMGConfig",
               "AutoModelForCausalLM": "emg_multilingual.EMGForCausalLM"}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


# --------------------------------------------------------------------------- #
#  binary-concrete relaxation (switchboard only; unchanged from v1)
# --------------------------------------------------------------------------- #
def gumbel_sigmoid(logits, tau, hard=False, training=True):
    if training:
        u = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
        noise = torch.log(u) - torch.log1p(-u)
        y = torch.sigmoid((logits + noise) / tau)
    else:
        y = torch.sigmoid(logits / tau)
    if hard:
        y = (y > 0.5).float() + y - y.detach()
    return y


# --------------------------------------------------------------------------- #
#  input-inferred parameters (UNCHANGED from v1 -- the live, language-encoding part)
# --------------------------------------------------------------------------- #
class InputInferredParameters(nn.Module):
    """Infer p in {0,1}^k from the input (masked mean-pool -> router -> k logits)."""

    def __init__(self, hidden_size, k=8, init_temp=1.0):
        super().__init__()
        self.k = k
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, k))
        self.default_logits = nn.Parameter(torch.zeros(k))
        self.register_buffer("temperature", torch.tensor(float(init_temp)))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self._last_logits = None

    def set_temperature(self, t):
        self.temperature.fill_(float(t))

    def infer_logits(self, x, attn_mask=None):
        if attn_mask is not None:
            m = attn_mask.unsqueeze(-1).to(x.dtype)
            pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        else:
            pooled = x.mean(1)
        return self.router(pooled)

    def forward(self, x, attn_mask=None, hard=False):
        logits = self.infer_logits(x, attn_mask)
        self._last_logits = logits
        return gumbel_sigmoid(logits, float(self.temperature), hard, training=self.training)

    def ig_penalty(self):
        if self._last_logits is None:
            return self.default_logits.new_zeros(())
        return ((self._last_logits - self.default_logits.unsqueeze(0)) ** 2).mean()


# --------------------------------------------------------------------------- #
#  controller: Merge + Move gates only (action head removed)
# --------------------------------------------------------------------------- #
class EMGController(nn.Module):
    def __init__(self, hidden_size, k_params, clamp=1.0):
        super().__init__()
        self.clamp = clamp
        self.W_in = nn.Linear(hidden_size, 2 * hidden_size)
        self.W_h = nn.Linear(hidden_size, 2 * hidden_size)
        self.param_to_gate = nn.Linear(k_params, 2 * hidden_size, bias=False)  # the LIVE pathway
        self._init()

    def _init(self):
        for lin in (self.W_in, self.W_h):
            nn.init.uniform_(lin.weight, -0.1, 0.1)
            nn.init.zeros_(lin.bias)
        nn.init.zeros_(self.param_to_gate.weight)        # p starts inert; learns under data pressure

    def forward(self, x, m_prev, p_lang):
        i_move, i_merge = torch.chunk(self.W_in(x), 2, dim=-1)
        h_move, h_merge = torch.chunk(self.W_h(m_prev), 2, dim=-1)
        b_move, b_merge = torch.chunk(self.param_to_gate(p_lang), 2, dim=-1)
        merge_gate = torch.clamp(i_merge * torch.sigmoid(h_merge + b_merge),
                                 -self.clamp, self.clamp)
        move_gate = torch.clamp(torch.sigmoid(i_move + b_move) * h_move,
                                -self.clamp, self.clamp)
        return merge_gate, move_gate


# --------------------------------------------------------------------------- #
#  deep, language-blind composer (unchanged)
# --------------------------------------------------------------------------- #
class _ComposeLayer(nn.Module):
    def __init__(self, h, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(h)
        self.ff = nn.Sequential(nn.Linear(h, 4 * h), nn.GELU(), nn.Linear(4 * h, h))
        self.drop = nn.Dropout(dropout)

    def forward(self, z):
        return z + self.drop(self.ff(self.norm(z)))


class UniversalComposer(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.combine = nn.Linear(2 * hidden_size, hidden_size)
        self.layers = nn.ModuleList([_ComposeLayer(hidden_size, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, a, b):
        z = self.combine(torch.cat([a, b], dim=-1))
        for layer in self.layers:
            z = layer(z)
        return self.norm(z)


# --------------------------------------------------------------------------- #
#  core: single continuous Merge+Move recurrence, always-on cache
# --------------------------------------------------------------------------- #
class EMGCore(nn.Module):
    def __init__(self, hidden_size, composer_layers, k_params, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.controller = EMGController(hidden_size, k_params)
        self.merge = UniversalComposer(hidden_size, composer_layers, dropout)
        self.push_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, p_lang, return_gates=False):
        B, T, H = x.shape
        m = x.new_zeros(B, H)        # working structure
        c = x.new_zeros(B, H)        # cache: substrate for Move
        outs, gates = [], ([] if return_gates else None)
        for t in range(T):
            xt = x[:, t]
            merge_gate, move_gate = self.controller(xt, m, p_lang)
            read = move_gate * c                       # Move: gated re-access of cache (always on)
            incoming = merge_gate + read               # element to merge: new (gate) + moved (read)
            m = self.merge(m, incoming)                # Merge into working structure
            # cache: write the merged-in material, consume what was moved (continuous push/pop)
            c = torch.tanh(torch.clamp(c + merge_gate - self.push_proj(read), -1.0, 1.0))
            outs.append(m)
            if return_gates:
                gates.append(torch.stack([merge_gate.abs().mean(-1),
                                          move_gate.abs().mean(-1)], dim=-1))  # (B,2)
        out = torch.stack(outs, dim=1)
        return out, (torch.stack(gates, dim=1) if return_gates else None)


# --------------------------------------------------------------------------- #
#  HF config + causal LM
# --------------------------------------------------------------------------- #
class EMGConfig(PretrainedConfig):
    model_type = "emg_ml"

    def __init__(self, vocab_size=32768, hidden_size=512, composer_layers=4,
                 k_params=8, dropout=0.1, temperature=1.0,
                 max_position_embeddings=1024, tie_word_embeddings=True,
                 bos_token_id=2, eos_token_id=3, pad_token_id=1, **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.composer_layers = composer_layers
        self.k_params = k_params
        self.dropout = dropout
        self.temperature = temperature
        self.max_position_embeddings = max_position_embeddings
        self.auto_map = {
            "AutoConfig": "emg_multilingual.EMGConfig",
            "AutoModelForCausalLM": "emg_multilingual.EMGForCausalLM",
        }
        super().__init__(tie_word_embeddings=tie_word_embeddings,
                         bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                         pad_token_id=pad_token_id, **kwargs)


class EMGForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = EMGConfig
    _tied_weights_keys = {"lm_head.weight": "wte.weight"}

    def __init__(self, config: EMGConfig):
        super().__init__(config)
        H = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, H)
        self.embed_drop = nn.Dropout(config.dropout)
        self.params = InputInferredParameters(H, config.k_params, config.temperature)
        self.core = EMGCore(H, config.composer_layers, config.k_params, config.dropout)
        self.lm_head = nn.Linear(H, config.vocab_size, bias=False)
        self.pad_id = config.pad_token_id
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, v):
        self.wte = v

    def get_output_embeddings(self):
        return self.lm_head

    def set_temperature(self, t):
        self.params.set_temperature(t)
        self.config.temperature = float(t)

    def parameter_penalty(self):
        """Input-Generalization prior over the inferred switchboard; add to loss."""
        return self.params.ig_penalty()

    @torch.no_grad()
    def infer_settings(self, input_ids, attention_mask=None):
        x = self.wte(input_ids)
        if attention_mask is None and self.pad_id is not None:
            attention_mask = (input_ids != self.pad_id).long()
        logits = self.params.infer_logits(x, attention_mask)
        return (torch.sigmoid(logits / float(self.params.temperature)) > 0.5).float()

    def switchboard_logits(self, input_ids, attention_mask=None):
        """Continuous (B, k) switchboard logits -- the hook for LiPs supervision."""
        x = self.wte(input_ids)
        if attention_mask is None and self.pad_id is not None:
            attention_mask = (input_ids != self.pad_id).long()
        return self.params.infer_logits(x, attention_mask)

    def forward(self, input_ids, attention_mask=None, labels=None,
                return_gates=False, **kwargs):
        x = self.embed_drop(self.wte(input_ids))
        if attention_mask is None and self.pad_id is not None:
            attention_mask = (input_ids != self.pad_id).long()
        p = self.params(x, attention_mask)                       # (B, k)
        out, gates = self.core(x, p, return_gates=return_gates)
        logits = self.lm_head(out)
        loss = None
        if labels is not None:
            sl = logits[:, :-1, :].contiguous()
            slab = labels[:, 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), slab.view(-1), ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits)


if __name__ == "__main__":
    cfg = EMGConfig(vocab_size=200, hidden_size=64, composer_layers=3, k_params=8)
    model = EMGForCausalLM(cfg)
    x = torch.randint(0, 200, (2, 16))
    o = model(input_ids=x)
    print("logits", tuple(o.logits.shape))
    print("inferred settings", model.infer_settings(x).tolist())
    print("ig penalty", float(model.parameter_penalty()))
