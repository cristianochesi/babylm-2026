#!/usr/bin/env python3
"""
HF-compatible single-language Hawk / RG-LRU model, for lm-eval.

Self-contained `trust_remote_code` modeling file. The building blocks are the
SAME code used at training time (De et al., 2024, Griffin/Hawk; arXiv:2402.19427),
so the per-language export state_dict maps 1:1 onto this module's parameters
(top-level attribute names wte / layers / norm_f / lm_head match the export keys
exactly -- no renaming, no transpose). Exposes the standard
forward(input_ids, labels=None) -> CausalLMOutputWithPast that lm-eval expects.

Register via config.json:
  "model_type": "hawk_rglru",
  "architectures": ["HawkForCausalLM"],
  "auto_map": {
    "AutoConfig": "modeling_hawk.HawkConfig",
    "AutoModelForCausalLM": "modeling_hawk.HawkForCausalLM",
    "AutoModelForSequenceClassification": "modeling_hawk.HawkForSequenceClassification"
  }
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import PreTrainedModel, PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutput, CausalLMOutputWithPast, SequenceClassifierOutput,
)

# transformers 4.x expects _tied_weights_keys as a LIST of target keys; 5.x expects a DICT
# {target: source} (5.13 hard-crashes on the list form in post_init). Adapt at import time.
_TF_MAJOR = int(transformers.__version__.split(".")[0])
_TIED_KEYS = ({"lm_head.weight": "wte.weight"} if _TF_MAJOR >= 5 else ["lm_head.weight"])

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * (x * norm)


def diag_linear_scan(a, b):
    """Inclusive parallel scan of h_t = a_t*h_{t-1} + b_t (h_0=0), diagonal affine.
    Hillis-Steele in real space: ceil(log2 T) vectorised passes, exact & stable
    (a in (0,1)), torch.compile-friendly (static shapes)."""
    T = a.shape[1]
    A, H = a, b
    d = 1
    while d < T:
        A_prev = torch.cat([A.new_ones(A.shape[0], d, A.shape[2]),  A[:, :-d]], dim=1)
        H_prev = torch.cat([H.new_zeros(H.shape[0], d, H.shape[2]), H[:, :-d]], dim=1)
        H = A * H_prev + H
        A = A * A_prev
        d *= 2
    return H


class RGLRU(nn.Module):
    """Real-Gated Linear Recurrent Unit (De et al., 2024)."""

    def __init__(self, width: int, c: float = 8.0, use_parallel_scan: bool = True):
        super().__init__()
        self.width = width
        self.c = c
        self.use_parallel_scan = use_parallel_scan
        self.input_gate = nn.Linear(width, width)
        self.recur_gate = nn.Linear(width, width)
        lam = torch.empty(width).uniform_(2.197, 6.907)
        self.log_lambda = nn.Parameter(lam)

    def forward(self, x):                       # x: (B, T, W)
        B, T, W = x.shape
        r = torch.sigmoid(self.recur_gate(x))
        i = torch.sigmoid(self.input_gate(x))
        log_a = -F.softplus(-self.log_lambda)
        log_a_t = self.c * r * log_a
        a_t = torch.exp(log_a_t)
        mult = torch.sqrt(torch.clamp(-torch.expm1(2.0 * log_a_t), min=1e-8))
        gated_x = mult * (i * x)
        if self.use_parallel_scan:
            return diag_linear_scan(a_t, gated_x)
        h = torch.zeros(B, W, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            h = a_t[:, t] * h + gated_x[:, t]
            outs.append(h)
        return torch.stack(outs, dim=1)


class RecurrentBlock(nn.Module):
    def __init__(self, d_model: int, d_rnn: int, conv_kernel: int = 4, rglru_c: float = 8.0):
        super().__init__()
        self.conv_kernel = conv_kernel
        self.in_gate = nn.Linear(d_model, d_rnn)
        self.in_recur = nn.Linear(d_model, d_rnn)
        self.conv = nn.Conv1d(d_rnn, d_rnn, conv_kernel, groups=d_rnn,
                              padding=conv_kernel - 1)
        self.rglru = RGLRU(d_rnn, rglru_c)
        self.out = nn.Linear(d_rnn, d_model)

    def forward(self, x):
        gate = F.gelu(self.in_gate(x))
        rec = self.in_recur(x).transpose(1, 2)
        rec = self.conv(rec)[..., : x.size(1)]
        rec = self.rglru(rec.transpose(1, 2))
        return self.out(gate * rec)


class MLPBlock(nn.Module):
    def __init__(self, d_model: int, expansion: int = 3):
        super().__init__()
        hidden = expansion * d_model
        self.gate = nn.Linear(d_model, hidden)
        self.up = nn.Linear(d_model, hidden)
        self.down = nn.Linear(hidden, d_model)

    def forward(self, x):
        return self.down(F.gelu(self.gate(x)) * self.up(x))


class HawkLayer(nn.Module):
    def __init__(self, d_model, d_rnn, conv_kernel, mlp_expansion, eps, rglru_c=8.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps)
        self.recur = RecurrentBlock(d_model, d_rnn, conv_kernel, rglru_c)
        self.norm2 = RMSNorm(d_model, eps)
        self.mlp = MLPBlock(d_model, mlp_expansion)

    def forward(self, x):
        x = x + self.recur(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HawkConfig(PretrainedConfig):
    model_type = "hawk_rglru"

    def __init__(self, vocab_size: int = 16384, n_layer: int = 12, n_embd: int = 768,
                 rnn_width: Optional[int] = None, conv_kernel: int = 4,
                 mlp_expansion: int = 3, rmsnorm_eps: float = 1e-6, rglru_c: float = 8.0,
                 max_position_embeddings: int = 1024, tie_word_embeddings: bool = True,
                 bos_token_id: int = 2, eos_token_id: int = 3, pad_token_id: int = 1,
                 **kwargs):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.rnn_width = rnn_width
        self.conv_kernel = conv_kernel
        self.mlp_expansion = mlp_expansion
        self.rmsnorm_eps = rmsnorm_eps
        self.rglru_c = rglru_c
        self.max_position_embeddings = max_position_embeddings
        self.auto_map = {
            "AutoConfig": "modeling_hawk.HawkConfig",
            # AutoModel = backbone only (last_hidden_state). Required by
            # finetune_token_classification.py, which wraps AutoModel in its own
            # PooledTokenClassifier instead of using AutoModelForTokenClassification.
            "AutoModel": "modeling_hawk.HawkModel",
            "AutoModelForCausalLM": "modeling_hawk.HawkForCausalLM",
            "AutoModelForSequenceClassification": "modeling_hawk.HawkForSequenceClassification",
        }
        super().__init__(tie_word_embeddings=tie_word_embeddings,
                         bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                         pad_token_id=pad_token_id, **kwargs)

    @property
    def d_rnn(self):
        return self.rnn_width if self.rnn_width is not None else self.n_embd

    # Standard HF configs expose the model width as `hidden_size`; HawkConfig
    # calls it `n_embd`. Generic pipeline code (e.g. the eval pipeline's
    # PooledTokenClassifier) reads config.hidden_size, so alias it to n_embd.
    # Read/write: from_pretrained may set hidden_size if it appears in a saved
    # config, and the setter keeps the two in sync instead of raising.
    @property
    def hidden_size(self):
        return self.n_embd

    @hidden_size.setter
    def hidden_size(self, value):
        self.n_embd = value


class HawkModel(PreTrainedModel):
    """Backbone only: embeddings -> HawkLayers -> final norm, returning
    `last_hidden_state`. No LM head, no task head.

    Exists because finetune_token_classification.py builds its own head on top of
    `AutoModel.from_pretrained(...)` and reads `outputs.last_hidden_state`;
    HawkForCausalLM returns logits, so it cannot serve that role. Attribute names
    (wte / layers / norm_f) are IDENTICAL to HawkForCausalLM, so a CausalLM
    checkpoint maps 1:1 onto this backbone with no renaming and nothing is newly
    initialised.
    """

    config_class = HawkConfig

    def __init__(self, config: HawkConfig):
        super().__init__(config)
        d_rnn = config.d_rnn
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([
            HawkLayer(config.n_embd, d_rnn, config.conv_kernel,
                      config.mlp_expansion, config.rmsnorm_eps, config.rglru_c)
            for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embd, config.rmsnorm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new):
        self.wte = new

    def forward(self, input_ids: torch.LongTensor,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> BaseModelOutput:
        x = self.wte(input_ids)
        for layer in self.layers:
            x = layer(x)
        return BaseModelOutput(last_hidden_state=self.norm_f(x))


class HawkForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = HawkConfig
    _tied_weights_keys = _TIED_KEYS                # dict on tf>=5, list on 4.x

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        lm = getattr(model, "lm_head", None)
        if lm is not None:
            if lm.weight.device.type == "meta":
                model.lm_head.weight = model.wte.weight
            elif lm.weight is not model.wte.weight:
                if torch.equal(lm.weight, model.wte.weight):
                    model.lm_head.weight = model.wte.weight  # harmless, just re-alias
                else:
                    raise ValueError(
                        "wte.weight and lm_head.weight differ despite tie_word_embeddings=True — "
                        "check the export script that produced this checkpoint "
                        "(likely a vocab-slicing mismatch between the two)."
                    )
        return model

    def __init__(self, config: HawkConfig):
        super().__init__(config)
        d_rnn = config.d_rnn
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([
            HawkLayer(config.n_embd, d_rnn, config.conv_kernel,
                      config.mlp_expansion, config.rmsnorm_eps, config.rglru_c)
            for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embd, config.rmsnorm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new):
        self.wte = new

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids: torch.LongTensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None,
                **kwargs) -> CausalLMOutputWithPast:
        x = self.wte(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1), ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits)


class HawkForSequenceClassification(PreTrainedModel):
    """
    Sequence-classification head on top of the SAME Hawk backbone used for
    causal LM. The backbone attribute names (wte / layers / norm_f) are
    IDENTICAL to HawkForCausalLM, so a CausalLM export state_dict maps 1:1 onto
    the backbone with no renaming. Only `score` is newly initialised, which is
    the expected behaviour when starting a fine-tuning run.

    The pooled representation is read from the hidden state at the last
    non-padding position (right padding, as produced by the BabyLM finetune
    tokenizer), matching the GPT-2 / Mamba sequence-classification convention.
    """

    config_class = HawkConfig

    def __init__(self, config: HawkConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        d_rnn = config.d_rnn
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([
            HawkLayer(config.n_embd, d_rnn, config.conv_kernel,
                      config.mlp_expansion, config.rmsnorm_eps, config.rglru_c)
            for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embd, config.rmsnorm_eps)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new):
        self.wte = new

    def forward(self, input_ids: torch.LongTensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None,
                **kwargs) -> SequenceClassifierOutput:
        x = self.wte(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        logits = self.score(x)                       # (B, T, num_labels)

        B, T = input_ids.shape[:2]
        # Index of the last real token per sequence (assumes right padding).
        if attention_mask is not None:
            last_idx = attention_mask.long().sum(-1) - 1
        elif self.config.pad_token_id is not None:
            last_idx = (input_ids != self.config.pad_token_id).int().sum(-1) - 1
        else:
            last_idx = torch.full((B,), T - 1, device=input_ids.device)
        last_idx = last_idx.clamp(min=0)
        pooled_logits = logits[torch.arange(B, device=input_ids.device), last_idx]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in (torch.long, torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                loss = (loss_fct(pooled_logits.squeeze(), labels.squeeze())
                        if self.num_labels == 1
                        else loss_fct(pooled_logits, labels))
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels),
                                labels.view(-1))
            else:  # multi_label_classification
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels.float())

        return SequenceClassifierOutput(loss=loss, logits=pooled_logits)