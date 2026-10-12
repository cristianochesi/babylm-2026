#!/usr/bin/env python3
"""
eMG-Sx -- expectation-based MG core with cue-based selective memory (Lewis & Vasishth 2005).
n_selective_states = 1 (eMG-S1: filler only) or 2 (eMG-S2: filler + binding).

CHANNEL-CAPACITY TIERS (the organizing principle, and the anti-starvation mechanism).
Pathways are differentiated by how much they can carry, matched to dependency range:
  * local encoder  : rich, width H -- full lexical / thematic detail, adjacency only.
  * filler bank    : mid, width val_fill -- compressed thematic-role POTENTIAL (survives to the
                     gap) + a percolated phi-subchannel for eager agreement. Long-range A-bar.
  * binding bank   : narrow, width val_bind (~32) -- phi only (person/number/gender/index).
                     The bottleneck PHYSICALLY cannot hold lexical detail, so it is forced to keep
                     the phi-abstraction binding needs. Discourse range, unsealed, bidirectional.
A narrow channel cannot do a wide channel's job, so capacity differentiation -> forced
specialization -> disjoint content -> every pathway is the unique carrier of some loss-relevant
signal -> none is redundant -> none starves. (Lossy-context surprisal: Futrell, Gibson & Levy 2020;
Levy 2008. Information bottleneck: Tishby et al.)

EAGER AGREEMENT (the [[the dog-s [that...]] run/*runs] case). On every write the filler bank also
PERCOLATES a gated copy of the new token's compressed value into the most-active slot (the active
head), so number/gender reaches the matrix-subject slot promptly, independent of the relative
clause. The verb later cues that slot and reads its phi -> agreement; partial phi-match from an
intervening noun = retrieval interference = agreement attraction (Wagers, Lau & Phillips 2009;
Lewis & Vasishth 2005).

BINDING (eMG-S2). A second bank, phi-cued, value-narrow, with an accessibility bias for the
pronoun (Condition A/B/C). It is NOT island-sealed (binding crosses islands) and is read
bidirectionally (forward + backward anaphora; Bianchi 2009). The exact A/B/C c-command predicate
is approximated here by a learned, p-shaped accessibility bias + recency; precise locality is the
natural target for weak structural supervision.

trust_remote_code: model_type "emg_sx", architectures ["EMGSxForCausalLM"],
auto_map -> emg_sx_multilingual.{EMGSxConfig,EMGSxForCausalLM}.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

ROBERTS_TABLE = {
    "head_dir_clausal": (1.0, 0.5, 1.0), "v2": (0.0, 1.0, 0.0),
    "null_subject": (0.0, 0.0, 1.0), "tense_morph": (1.0, 1.0, 0.0),
    "sv_agreement": (1.0, 1.0, 0.0), "filler_gap": (1.0, 1.0, 0.0),
    "head_dir_dp": (0.5, 0.5, 0.0), "def_article": (1.0, 1.0, 0.0),
    "number_morph": (1.0, 1.0, 0.0), "classifier": (0.0, 0.0, 1.0),
    "binding_domain": (0.0, 0.0, 1.0), "rc_position": (1.0, 1.0, 0.0),
}
DEFAULT_SUPERVISE = ["v2", "filler_gap"]


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__(); self.g = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return self.g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def gumbel_sigmoid(logits, tau, hard=False, training=True):
    if training:
        u = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
        y = torch.sigmoid((logits + torch.log(u) - torch.log1p(-u)) / tau)
    else:
        y = torch.sigmoid(logits / tau)
    if hard:
        y = (y > 0.5).float() + y - y.detach()
    return y


class InputInferredParameters(nn.Module):
    def __init__(self, hidden, k=8, init_temp=1.0, logit_cap=6.0):
        super().__init__()
        self.router = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, k))
        self.default_logits = nn.Parameter(torch.zeros(k))
        self.register_buffer("temperature", torch.tensor(float(init_temp)))
        self.logit_cap = float(logit_cap)
        nn.init.zeros_(self.router[-1].weight); nn.init.zeros_(self.router[-1].bias)
        self._last_logits = None
    def set_temperature(self, t): self.temperature.fill_(float(t))
    def forward(self, x, attn_mask=None, hard=False):
        if attn_mask is not None:
            m = attn_mask.unsqueeze(-1).to(x.dtype); pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        else:
            pooled = x.mean(1)
        raw = self.router(pooled)
        logits = self.logit_cap * torch.tanh(raw / self.logit_cap) if self.logit_cap > 0 else raw
        self._last_logits = logits
        return gumbel_sigmoid(logits, float(self.temperature), hard, self.training)
    def ig_penalty(self):
        if self._last_logits is None:
            return self.default_logits.new_zeros(())
        # default_logits is a FREE parameter while _last_logits is tanh-capped to +/-logit_cap.
        # If default_logits drifts past the cap the squared gap (and its gradient) explode and
        # sharpen the router -> reads sharpen -> the recurrence tips. Bound BOTH sides to the same
        # range so the penalty stays finite and cannot self-amplify.
        dl = self.default_logits
        if self.logit_cap > 0:
            dl = self.logit_cap * torch.tanh(dl / self.logit_cap)
        return ((self._last_logits - dl.unsqueeze(0)) ** 2).mean()


class LocalEncoder(nn.Module):
    """Rich short-range: causal conv (adjacency) + gated MLP, FiLM by p. No long-range carry."""
    def __init__(self, hidden, kernel=4, k_params=8):
        super().__init__()
        self.conv = nn.Conv1d(hidden, hidden, kernel, groups=hidden, padding=kernel - 1)
        self.n1 = RMSNorm(hidden); self.n2 = RMSNorm(hidden)
        self.gate = nn.Linear(hidden, hidden); self.up = nn.Linear(hidden, hidden)
        self.down = nn.Linear(hidden, hidden)
        self.film = nn.Linear(k_params, hidden, bias=False); nn.init.normal_(self.film.weight, std=0.02)
    def forward(self, x, p):
        T = x.size(1)
        c = self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2)
        c = self.n1(x + c)
        h = self.down(F.gelu(self.gate(c)) * self.up(c)) + self.film(p).unsqueeze(1)
        return self.n2(c + h)


class CueMemory(nn.Module):
    """Content-addressable cue-based bank with an explicit VALUE width (the channel capacity).
    Stored value width = val_dim; a narrow val_dim is the information bottleneck that forces the
    bank to keep only what fits. Optional eager-agreement percolation into the most-active slot."""
    def __init__(self, in_dim, n_slots, cue_dim, val_dim, decay=0.95, base_w=0.3,
                 temp=None, percolate=False, act_cap=1e3):
        super().__init__()
        self.S, self.dk, self.vd = n_slots, cue_dim, val_dim
        self.decay = float(decay); self.base_w = float(base_w)
        self.act_cap = float(act_cap)                       # upper bound on the recency accumulator
        self.temp = float(temp) if temp else math.sqrt(cue_dim)
        self.val_in = nn.Linear(in_dim, val_dim)            # compress input -> capacity-limited value
        self.key_w = nn.Linear(val_dim, cue_dim, bias=False)
        self.cue_q = nn.Linear(cue_dim, cue_dim, bias=False)
        self.val_norm = RMSNorm(val_dim)
        self.slot_key = nn.Parameter(torch.randn(n_slots, cue_dim))
        self.percolate = percolate
        if percolate:
            self.perc_gate = nn.Linear(val_dim, 1)
    def init_state(self, B, device, dtype=None):
        # The recurrent state is kept in fp32 (ignoring the autocast dtype): a 256-step scan that
        # accumulates `act` and feeds it through log() loses too much in bf16's 8-bit mantissa.
        f32 = torch.float32
        bank = torch.zeros(B, self.S, self.vd, device=device, dtype=f32)
        keys = torch.zeros(B, self.S, self.dk, device=device, dtype=f32)
        act = torch.linspace(0.0, 0.05, self.S, device=device, dtype=f32).expand(B, self.S).contiguous()
        return bank, keys, act
    def read(self, cue, bank, keys, act, access_bias=None):
        # Score/softmax in fp32 (state is already fp32). Autocast is disabled here so the einsum
        # and softmax over slots are computed at full precision -- this is the numerically fragile
        # part of the scan.
        with torch.autocast(device_type=bank.device.type, enabled=False):
            cue = cue.float()
            if access_bias is not None:
                access_bias = access_bias.float()
            q = self.cue_q(cue)
            score = torch.einsum("bd,bsd->bs", q, keys + self.slot_key.float()) / self.temp
            # Recency is a STOP-GRADIENT bias: it steers which slot is read but does not send
            # gradient back through the depth-256 `act` recurrence (that path was the exploding
            # Jacobian). log1p keeps the gradient of any downstream use of `a_tot` bounded.
            a_tot = score + self.base_w * torch.log1p(act.detach().clamp_min(0.0))
            if access_bias is not None:
                a_tot = a_tot + access_bias
            w = F.softmax(a_tot, dim=-1)
            ret = torch.einsum("bs,bsv->bv", w, bank)
        return ret, w
    def write(self, src, bank, keys, act):
        # bank/keys/act are fp32; compute the whole write in fp32 so the running state never
        # touches bf16. val_in/key_w/perc_gate inputs are upcast here.
        with torch.autocast(device_type=bank.device.type, enabled=False):
            c = self.val_norm(self.val_in(src.float()))
            wnew = F.softmax(-act / max(self.temp * 0.25, 1e-3), dim=-1)     # new constituent: least active
            bank = bank * (1 - wnew.unsqueeze(-1)) + wnew.unsqueeze(-1) * c.unsqueeze(1)
            k = self.key_w(c)
            keys = keys * (1 - wnew.unsqueeze(-1)) + wnew.unsqueeze(-1) * k.unsqueeze(1)
            if self.percolate:                                              # eager agreement transmission
                whead = F.softmax(act / max(self.temp * 0.25, 1e-3), dim=-1) # active head: most active
                g = torch.sigmoid(self.perc_gate(c)) * 0.5
                upd = (g * whead).unsqueeze(-1)
                bank = bank + upd * (c.unsqueeze(1) - bank)
                keys = keys + upd * (k.unsqueeze(1) - keys)
            # cap the recency accumulator: unbounded growth is harmless for the softmaxes above but
            # cheap insurance against a pathological run inflating it toward fp32's ceiling.
            act = (act * self.decay + wnew).clamp(max=self.act_cap)
        return bank, keys, act
    @staticmethod
    def discharge(w, gate, act):
        return act * (1 - gate * w)


class EMGSxConfig(PretrainedConfig):
    model_type = "emg_sx"
    def __init__(self, vocab_size=32768, hidden_size=512, conv_kernel=4, n_categories=8,
                 n_selective_states=2, k_params=8, dropout=0.1, gumbel_temperature=1.0,
                 n_slots_fill=24, cue_dim_fill=96, val_dim_fill=96,
                 n_slots_bind=16, cue_dim_bind=32, val_dim_bind=32,
                 mem_decay=0.95, mem_base_w=0.3, mcp_bias=1.0,
                 logit_cap=6.0, param_label_smooth=0.05, param_reg=3e-3, lambda_param_sup=0.0,
                 tbptt_steps=32, act_cap=1e3,
                 max_position_embeddings=1024, tie_word_embeddings=True,
                 bos_token_id=2, eos_token_id=3, pad_token_id=1, **kw):
        self.vocab_size = vocab_size; self.hidden_size = hidden_size
        self.conv_kernel = conv_kernel; self.n_categories = n_categories
        self.n_selective_states = n_selective_states; self.k_params = k_params
        self.dropout = dropout; self.gumbel_temperature = gumbel_temperature
        self.n_slots_fill = n_slots_fill; self.cue_dim_fill = cue_dim_fill; self.val_dim_fill = val_dim_fill
        self.n_slots_bind = n_slots_bind; self.cue_dim_bind = cue_dim_bind; self.val_dim_bind = val_dim_bind
        self.mem_decay = mem_decay; self.mem_base_w = mem_base_w; self.mcp_bias = mcp_bias
        self.logit_cap = logit_cap; self.param_label_smooth = param_label_smooth
        self.param_reg = param_reg; self.lambda_param_sup = lambda_param_sup
        self.tbptt_steps = int(tbptt_steps); self.act_cap = float(act_cap)
        self.max_position_embeddings = max_position_embeddings
        self.auto_map = {"AutoConfig": "emg_sx_multilingual.EMGSxConfig",
                         "AutoModelForCausalLM": "emg_sx_multilingual.EMGSxForCausalLM"}
        super().__init__(tie_word_embeddings=tie_word_embeddings, bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kw)


class EMGSxForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = EMGSxConfig
    _tied_weights_keys = {"lm_head.weight": "wte.weight"}
    def __init__(self, cfg: EMGSxConfig):
        super().__init__(cfg)
        assert cfg.n_selective_states in (1, 2)
        H, K = cfg.hidden_size, cfg.n_categories
        self.use_bind = cfg.n_selective_states >= 2
        self.wte = nn.Embedding(cfg.vocab_size, H)
        self.embed_drop = nn.Dropout(cfg.dropout)
        self.params = InputInferredParameters(H, cfg.k_params, cfg.gumbel_temperature, cfg.logit_cap)
        self.local = LocalEncoder(H, cfg.conv_kernel, cfg.k_params)
        # filler bank (mid capacity, eager-agreement percolation)
        self.tbptt = int(getattr(cfg, "tbptt_steps", 32))          # truncated-BPTT window (0 = full)
        self.mem_f = CueMemory(H, cfg.n_slots_fill, cfg.cue_dim_fill, cfg.val_dim_fill,
                               cfg.mem_decay, cfg.mem_base_w, percolate=True,
                               act_cap=getattr(cfg, "act_cap", 1e3))
        self.cue_f = nn.Linear(H, cfg.cue_dim_fill)
        self.up_f = nn.Linear(cfg.val_dim_fill, H)
        self.disc_proj = nn.Linear(H, 1)
        self.access_f = nn.Linear(cfg.k_params, cfg.n_slots_fill, bias=False); nn.init.zeros_(self.access_f.weight)
        if self.use_bind:                                          # binding bank (narrow, phi only)
            self.mem_b = CueMemory(H, cfg.n_slots_bind, cfg.cue_dim_bind, cfg.val_dim_bind,
                                   cfg.mem_decay, cfg.mem_base_w, percolate=False,
                                   act_cap=getattr(cfg, "act_cap", 1e3))
            self.cue_b = nn.Linear(H, cfg.cue_dim_bind)
            self.up_b = nn.Linear(cfg.val_dim_bind, H)
            self.bind_gate = nn.Linear(H, 1)                       # is this a pronoun? gate the read
            self.access_b = nn.Linear(cfg.k_params, cfg.n_slots_bind, bias=False); nn.init.zeros_(self.access_b.weight)
        self.out_norm = RMSNorm(H)
        self.cat_prior = nn.Linear(H, K); self.cat_emb = nn.Embedding(K, H)
        self.lm_head = nn.Linear(H, cfg.vocab_size, bias=False)
        self.K = K; self.pad_id = cfg.pad_token_id
        nn.init.constant_(self.disc_proj.bias, float(cfg.mcp_bias))
        self._aux = {}
        self.register_buffer("param_targets", torch.zeros(0), persistent=False)
        self.register_buffer("param_sup_mask", torch.zeros(0), persistent=False)
        self._lang_to_idx = {}
        self.post_init()
    def get_input_embeddings(self): return self.wte
    def set_input_embeddings(self, v): self.wte = v
    def get_output_embeddings(self): return self.lm_head
    def set_temperature(self, t): self.params.set_temperature(t); self.config.gumbel_temperature = float(t)
    def set_param_targets(self, langs, table=None, supervise=None, bit_assignment=None):
        table = table or ROBERTS_TABLE; supervise = supervise or DEFAULT_SUPERVISE
        if bit_assignment is None: bit_assignment = {n: i for i, n in enumerate(supervise)}
        k = self.config.k_params; self._lang_to_idx = {lg: i for i, lg in enumerate(langs)}
        tgt = torch.zeros(len(langs), k); mask = torch.zeros(k)
        for name in supervise:
            bit = bit_assignment[name]; mask[bit] = 1.0
            for li in range(len(langs)): tgt[li, bit] = float(table[name][li])
        dev = self.wte.weight.device
        self.param_targets = tgt.to(dev); self.param_sup_mask = mask.to(dev)
        return dict(self._lang_to_idx)
    def _mos_logmarginal(self, h, pi_log):
        B, H = h.shape
        comp = h.unsqueeze(1) + self.cat_emb.weight.unsqueeze(0)
        logits = self.lm_head(comp.reshape(B * self.K, H)).reshape(B, self.K, -1)
        return torch.logsumexp(pi_log.unsqueeze(-1) + F.log_softmax(logits, -1), dim=1)
    def aux_loss(self):
        if not self._aux: return self.wte.weight.new_zeros(())
        c = self.config
        return (c.param_reg * self._aux.get("param", 0.0)
                + getattr(c, "lambda_param_sup", 0.0) * self._aux.get("param_sup", 0.0))
    def aux_components(self):
        out = {}
        for k in ("param", "param_sup"):
            v = self._aux.get(k)
            if v is not None: out[k] = float(v.detach()) if torch.is_tensor(v) else float(v)
        return out
    def diagnostics(self):
        d = getattr(self, "_diag", {})
        return {k: (v.detach().item() if torch.is_tensor(v) else v) for k, v in d.items()}
    def forward(self, input_ids, attention_mask=None, labels=None, lang_ids=None, **kw):
        x = self.embed_drop(self.wte(input_ids)); B, T, H = x.shape
        if attention_mask is None and self.pad_id is not None:
            attention_mask = (input_ids != self.pad_id).long()
        p = self.params(x, attention_mask)
        acc_f = self.access_f(p)
        acc_b = self.access_b(p) if self.use_bind else None
        loc = self.local(x, p)
        bf, kf, af = self.mem_f.init_state(B, x.device, x.dtype)
        if self.use_bind: bb, kb, ab = self.mem_b.init_state(B, x.device, x.dtype)
        m = attention_mask if attention_mask is not None else torch.ones(B, T, device=x.device)
        logits = []
        d_disc = x.new_zeros(()); d_ef = x.new_zeros(()); d_eb = x.new_zeros(())
        for t in range(T):
            lt = loc[:, t]
            ret_f, wf = self.mem_f.read(self.cue_f(lt), bf, kf, af, acc_f)        # filler: argument / gap
            disc = torch.sigmoid(self.disc_proj(lt))
            inj = self.up_f(ret_f) * disc
            if self.use_bind:
                ret_b, wb = self.mem_b.read(self.cue_b(lt), bb, kb, ab, acc_b)     # binding: antecedent
                bgate = torch.sigmoid(self.bind_gate(lt))
                inj = inj + self.up_b(ret_b) * bgate
            h_out = self.out_norm(lt + inj)
            af = self.mem_f.discharge(wf, disc, af)
            bf, kf, af = self.mem_f.write(lt, bf, kf, af)                          # theta potential + percolate phi
            if self.use_bind:
                bb, kb, ab = self.mem_b.write(lt, bb, kb, ab)                      # narrow phi
            # Truncated BPTT: sever the graph at fixed windows so the backward pass through this
            # multiplicative memory scan can never span more than `tbptt` steps (the depth-256
            # product was the exploding-gradient source). Forward values are unchanged.
            if self.tbptt and (t + 1) % self.tbptt == 0 and (t + 1) < T:
                bf, kf, af = bf.detach(), kf.detach(), af.detach()
                if self.use_bind:
                    bb, kb, ab = bb.detach(), kb.detach(), ab.detach()
            pi_log = F.log_softmax(self.cat_prior(h_out), -1)
            logits.append(self._mos_logmarginal(h_out, pi_log))
            with torch.no_grad():
                msk = m[:, t]
                d_disc = d_disc + (disc.squeeze(-1) * msk).sum()
                d_ef = d_ef - ((wf.clamp_min(1e-9).log() * wf).sum(-1) * msk).sum()
                if self.use_bind: d_eb = d_eb - ((wb.clamp_min(1e-9).log() * wb).sum(-1) * msk).sum()
        logits = torch.stack(logits, 1)
        self._aux = {"param": self.params.ig_penalty()}
        vc = m.sum().clamp(min=1.0)
        self._diag = {"discharge_rate": (d_disc / vc).detach(), "fill_entropy": (d_ef / vc).detach()}
        if self.use_bind: self._diag["bind_entropy"] = (d_eb / vc).detach()
        if (getattr(self.config, "lambda_param_sup", 0.0) > 0 and lang_ids is not None
                and self.param_targets.numel() > 0):
            s = float(self.config.param_label_smooth)
            tgt = self.param_targets[lang_ids] * (1 - 2 * s) + s
            bce = F.binary_cross_entropy_with_logits(self.params._last_logits, tgt, reduction="none")
            msk = self.param_sup_mask.unsqueeze(0)
            self._aux["param_sup"] = (bce * msk).sum() / (msk.sum() * B).clamp(min=1.0)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                                   ignore_index=-100) + self.aux_loss()
        return CausalLMOutputWithPast(loss=loss, logits=logits)


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = EMGSxConfig(vocab_size=200, hidden_size=64, conv_kernel=4, n_categories=8,
                      n_selective_states=2, k_params=8,
                      n_slots_fill=12, cue_dim_fill=24, val_dim_fill=32,
                      n_slots_bind=8, cue_dim_bind=16, val_dim_bind=16, lambda_param_sup=0.1)
    model = EMGSxForCausalLM(cfg)
    lang_map = model.set_param_targets(["eng", "nld", "zho"], supervise=["v2", "filler_gap"])
    ids = torch.randint(0, 200, (3, 20))
    lang_ids = torch.tensor([lang_map[l] for l in ("eng", "nld", "zho")])
    with torch.no_grad(): model.wte.weight.mul_(8.0)
    out = model(input_ids=ids, labels=ids, lang_ids=lang_ids)
    print("logits", tuple(out.logits.shape), "| loss", round(float(out.loss.detach()), 4),
          "| finite", bool(torch.isfinite(out.loss).item()))
    print("value widths -> local:", cfg.hidden_size, " filler:", cfg.val_dim_fill, " binding:", cfg.val_dim_bind)
    print("diag:", {k: round(v, 4) for k, v in model.diagnostics().items()})
    out.loss.backward()
    def g(p): return f"{p.grad.norm().item():.2e}" if p is not None and p.grad is not None else None
    print("FILLER  bank grads:", {"cue": g(model.cue_f.weight), "key": g(model.mem_f.key_w.weight),
                                  "percolate": g(model.mem_f.perc_gate.weight)})
    print("BINDING bank grads:", {"cue": g(model.cue_b.weight), "key": g(model.mem_b.key_w.weight),
                                  "val_in": g(model.mem_b.val_in.weight)})