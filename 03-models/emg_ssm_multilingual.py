#!/usr/bin/env python3
"""
Phase-segmented selective-SSM neural e-MG -- ROUTE 1 + MAGNITUDE CONTROL.

This is the Route-1 model with reliable numerical stabilization. The Route-1 run woke the
phase pathway (succ went from a flat 0.000 to violently active), which exposed that NOTHING
in the recurrence was magnitude-bounded: pre-clip grad norms hit 1e15 on merge/read_out/
dt_proj/action_head, the SSM state wound up (succ -> 3.7), the spine read summed those large
states and scaled with depth, the router logits ran to +/-inf (pen -> 20), and ce re-spiked
to ln(V). grad_clip=1.0 caught the blowups but a 1e15 -> 1.0 rescale is a unit step in a
meaningless direction, so the trajectory oscillated and BLiMP sat at chance, non-monotone.

STABILIZATION (all on by default; each bounds one previously-unbounded quantity):
  (S1) SSM state windup. The integrator pole a = exp(dt*A) can approach 1 (A_log drifting
       very negative -> infinite memory), and a single large dt makes a huge one-step write.
       Fix: floor the decay  A = -exp(A_log.clamp(min=a_log_min))  so the steady state
       ~(B*x)/|A| stays bounded, and cap  dt = softplus(...).clamp(max=dt_max)  so per-step
       writes can't spike. This is what stops succ (= active-slice state norm) from blowing up.
  (S2) Spine read. The LM/controller read summed ancestor states and scaled with depth, so a
       deep open (dep 4-5) multiplied magnitude by ~5. Fix: read the depth-normalized MEAN of
       the path (divide by the number of active slices), and RMSNorm the read before it enters
       merge. This bounds what feeds both the marginal and the action head -- the direct cause
       of the merge/read_out/action_head grad explosion and the ce re-spike. Write/seal is
       UNCHANGED, so islands stay mechanical.
  (S3) Router logits. BCE-to-bits has no finite optimum at 0/1 targets, so it pushes logits to
       +/-inf (pen climbs to 20). Fix: tanh-cap the router logits to (-logit_cap, logit_cap)
       AND label-smooth the BCE targets to {s, 1-s}. The smoothed optimum sits at finite logits
       (+/- ln((1-s)/s)) well inside the cap, so logits and pen stay bounded with no runaway.

grad_clip=1.0 remains as a final safety net but should now rarely fire; watch the grad-path
probe -- norms should stay O(1) instead of O(1e15).

Recommended run-side changes (see sbatch): lower lambda_param_sup (0.5 over-drove psup to 0
by it~750) and lambda_load (it forced the deep opens that triggered the blowup); the goal is
bits that are decodable AND used, not psup=0.

Set spine_read=False, param_film=False, read_norm=False, logit_cap=0 to recover the unbounded
Route-1 behavior for ablation. HF-native trust_remote_code modeling file.
  "model_type": "ssm_emg_ml", "architectures": ["EMGSSMForCausalLM"],
  "auto_map": {"AutoConfig": "emg_ssm_multilingual.EMGSSMConfig",
               "AutoModelForCausalLM": "emg_ssm_multilingual.EMGSSMForCausalLM"}
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


# Roberts (2019) hierarchy across ENG, NLD, ZHO. (eng, nld, zho); + -> 1, - -> 0, mixed -> 0.5.
# RANK 2 over these 3 languages: AXIS 1 {eng,nld} vs {zho} (10 functional-morphology params),
# AXIS 2 {nld} vs {eng,zho} (verb-second). Identify >2 params only by adding languages.
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
    """Portable RMSNorm (no torch-version dependency)."""
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d)); self.eps = eps

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


# --------------------------------------------------------------------------- #
#  input-inferred binary-concrete parameters (the multilingual switchboard)
# --------------------------------------------------------------------------- #
class InputInferredParameters(nn.Module):
    def __init__(self, hidden, k=8, init_temp=1.0, logit_cap=6.0):
        super().__init__()
        self.router = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, k))
        self.default_logits = nn.Parameter(torch.zeros(k))
        self.register_buffer("temperature", torch.tensor(float(init_temp)))
        self.logit_cap = float(logit_cap)                      # (S3) bound router logits
        nn.init.zeros_(self.router[-1].weight); nn.init.zeros_(self.router[-1].bias)
        self._last_logits = None

    def set_temperature(self, t): self.temperature.fill_(float(t))

    def forward(self, x, attn_mask=None, hard=False):
        if attn_mask is not None:
            m = attn_mask.unsqueeze(-1).to(x.dtype)
            pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        else:
            pooled = x.mean(1)
        raw = self.router(pooled)
        # (S3) tanh-cap: logits live in (-cap, cap); with label-smoothed BCE the optimum is
        # at finite logits inside this range, so nothing runs to +/-inf.
        logits = self.logit_cap * torch.tanh(raw / self.logit_cap) if self.logit_cap > 0 else raw
        self._last_logits = logits
        return gumbel_sigmoid(logits, float(self.temperature), hard, self.training)

    def ig_penalty(self):
        if self._last_logits is None:
            return self.default_logits.new_zeros(())
        return ((self._last_logits - self.default_logits.unsqueeze(0)) ** 2).mean()


# --------------------------------------------------------------------------- #
#  phase-segmented selective SSM (Move): hard seal + bounded state + bounded read
# --------------------------------------------------------------------------- #
class PhaseSelectiveSSM(nn.Module):
    def __init__(self, hidden, d_state=8, n_phase=6, k_params=8, param_film=True,
                 spine_read=True, read_norm=True, a_log_min=math.log(0.02), dt_max=2.0,
                 seal_read=True, seal_init=0.85):
        super().__init__()
        self.H, self.N, self.P = hidden, d_state, n_phase
        self.spine_read = spine_read
        # (S4) GRADED PARAMETER-DRIVEN READ SEAL. The uniform spine read folds every ancestor
        # into the read at equal weight -> the parent is fully readable from inside a child phase
        # (the "island seal FAIL"). Replace the flat mean with a depth-graded weighted mean whose
        # decay is a per-sequence scalar `seal` in (0,1) produced from the multilingual parameter
        # vector p:  weight(slice) = seal ** (dist-below-active).  seal->1 recovers the old uniform
        # read (parent fully leaks; no island); seal->0 isolates the active slice (hard seal; the
        # parent's filler cannot be extracted across the boundary). Because seal reads p, a +movement
        # language (filler_gap=1: eng,nld) can learn to seal harder than a wh-in-situ one (zho),
        # tying islandhood to the typological parameter -- one knob serving both parameterization
        # and island-selectivity, degrading gracefully to both prior modes.
        self.seal_read = bool(seal_read)
        if self.seal_read:
            self.seal_from_p = nn.Linear(k_params, 1, bias=False)
            nn.init.normal_(self.seal_from_p.weight, std=0.02)
            b0 = math.log(seal_init / (1.0 - seal_init))          # logit(seal_init)
            self.seal_bias = nn.Parameter(torch.tensor(float(b0)))
        self._last_seal = None            # (B,1) cached each step for probes / tracing
        self._last_h_read = None          # (B,H,N) graded state read (pre C-projection)
        self._last_read_w = None          # (B,P) normalized read mass per phase slice
        self.a_log_min = float(a_log_min)      # (S1) floor |A| -> bound state windup
        self.dt_max = float(dt_max)            # (S1) cap dt   -> bound per-step write
        self.dt_proj = nn.Linear(hidden, hidden)
        self.B_proj = nn.Linear(hidden, d_state)
        self.C_proj = nn.Linear(hidden, d_state)
        self.x_in = nn.Linear(hidden, hidden)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()
                                            .repeat(hidden, 1)))
        self.read_out = nn.Linear(hidden, hidden)
        self.read_norm = RMSNorm(hidden) if read_norm else None   # (S2) bound the read
        if param_film:
            self.dt_film = nn.Linear(k_params, hidden, bias=False)
            self.B_film = nn.Linear(k_params, d_state, bias=False)
            self.C_film = nn.Linear(k_params, d_state, bias=False)
            for mod in (self.dt_film, self.B_film, self.C_film):
                nn.init.normal_(mod.weight, std=0.02)
        else:
            self.dt_film = self.B_film = self.C_film = None
        self._film_p = None

    def init_state(self, B, device, dtype):
        buf = torch.zeros(B, self.P, self.H, self.N, device=device, dtype=dtype)
        depth = torch.zeros(B, self.P, device=device, dtype=dtype); depth[:, 0] = 1.0
        return buf, depth

    @staticmethod
    def _hard(depth):
        idx = depth.argmax(-1, keepdim=True)
        hard = torch.zeros_like(depth).scatter_(-1, idx, 1.0)
        return hard + depth - depth.detach()

    def seal_from_params(self, p):
        """Per-sequence seal scalar in (0,1). p:(B,k)->(B,1). Higher = softer (parent leaks);
        lower = harder (active-slice isolation). Constant across time (p is sequence-pooled)."""
        if not self.seal_read:
            return None
        if p is None:
            return torch.sigmoid(self.seal_bias).view(1, 1)
        return torch.sigmoid(self.seal_bias + self.seal_from_p(p))

    def _graded_state_read(self, buf, depth, d, p):
        """Depth-graded weighted mean of the open path (active + attenuated ancestors), used by
        step() and by the island probe. seal->1 recovers the uniform spine read (parent leaks);
        seal->0 isolates the active slice (hard seal). Sets _last_seal / _last_read_w / _last_h_read."""
        if self.spine_read:
            path = torch.flip(torch.cumsum(torch.flip(depth, [-1]), -1), [-1]).clamp(max=1.0)  # (B,P)
            if self.seal_read:
                seal = self.seal_from_params(p)                      # (B,1) in (0,1)
                if seal.size(0) == 1 and depth.size(0) > 1:
                    seal = seal.expand(depth.size(0), 1)
                self._last_seal = seal.detach()
                lvl = torch.arange(self.P, device=depth.device, dtype=depth.dtype)      # (P,)
                active_lvl = (depth * lvl).sum(-1, keepdim=True)     # (B,1) soft active frontier
                dist = (active_lvl - lvl[None, :]).clamp(min=0.0)    # (B,P) depth below active
                w = path * (seal ** dist)                            # active->1, ancestors decayed
            else:
                self._last_seal = None
                w = path                                            # uniform (legacy spine read)
            wden = w.sum(1).clamp(min=1e-4)                          # (B,)
            h_read = (w[:, :, None, None] * buf).sum(1) / wden[:, None, None]   # (B,H,N)
            self._last_read_w = (w / wden[:, None]).detach()        # (B,P) normalized read mass
        else:
            self._last_seal = None
            h_read = (d[:, :, None, None] * buf).sum(1)
            self._last_read_w = d.detach()
        self._last_h_read = h_read.detach()                          # (B,H,N) pre-projection read
        return h_read

    def step(self, buf, depth, x, action):
        d = self._hard(depth)                               # (B,P) active slice (hard fwd)
        A = -torch.exp(self.A_log.clamp(min=self.a_log_min))  # (S1) floored decay (H,N)
        dt_pre = self.dt_proj(x)
        Bw = self.B_proj(x); Cw = self.C_proj(x)
        p = self._film_p
        if p is not None and self.dt_film is not None:
            dt_pre = dt_pre + self.dt_film(p)
            Bw = Bw + self.B_film(p)
            Cw = Cw + self.C_film(p)
        dt = F.softplus(dt_pre).clamp(max=self.dt_max)      # (S1) bounded write rate (B,H)
        xin = self.x_in(x)
        a = torch.exp(dt[:, :, None] * A[None])             # (B,H,N) in (0,1)
        write = (dt[:, :, None] * Bw[:, None, :]) * xin[:, :, None]   # (B,H,N)
        upd = a[:, None] * buf + write[:, None]             # (B,P,H,N)
        dw = d[:, :, None, None]
        buf = (1 - dw) * buf + dw * upd                     # freeze non-active slices (seal)

        # (S2)+(S4) depth-graded read of the open path (see _graded_state_read). Write stays sealed
        # (buf freeze above is UNCHANGED); only the READ is depth-attenuated by the seal.
        h_read = self._graded_state_read(buf, depth, d, p)
        read = self.read_out((Cw[:, None, :] * h_read).sum(-1))
        if self.read_norm is not None:
            read = self.read_norm(read)                     # (S2) bounded read -> stable merge

        up = torch.zeros_like(depth); up[:, 1:] = depth[:, :-1]; up[:, -1] = up[:, -1] + depth[:, -1]
        down = torch.zeros_like(depth); down[:, :-1] = depth[:, 1:]; down[:, 0] = down[:, 0] + depth[:, 0]
        new_depth = action[:, 0:1] * depth + action[:, 1:2] * up + action[:, 2:3] * down

        child_norm = (d[:, :, None, None] * buf).pow(2).sum((1, 2, 3)).sqrt()   # (B,)
        success = (action[:, 2] * child_norm).mean()
        reset = (action[:, 2:3] * d).unsqueeze(-1).unsqueeze(-1)
        buf = (1 - reset) * buf
        return buf, new_depth, read, success


# --------------------------------------------------------------------------- #
#  controller: parameters bias the phase action (RNNG-style 3-way)
# --------------------------------------------------------------------------- #
class PhaseController(nn.Module):
    def __init__(self, hidden, k_params):
        super().__init__()
        self.in_proj = nn.Linear(2 * hidden, hidden)
        self.action_head = nn.Linear(hidden, 3)
        self.param_to_action = nn.Linear(k_params, 3, bias=False)
        nn.init.normal_(self.param_to_action.weight, std=0.02)
        # (S5) SOFT STAY CURRICULUM. A large positive bias on the STAY logit keeps the phase
        # pathway quiet early (no violent open/close before the SSM is conditioned), then is
        # annealed to 0 to release nesting -- the graceful version of force_stay. The training
        # loop calls set_stay_bias(schedule(step)); 0.0 (default) leaves behavior unchanged.
        self.register_buffer("stay_bias", torch.tensor(0.0))

    def set_stay_bias(self, v):
        self.stay_bias.fill_(float(v))

    def forward(self, x, summary, p_lang, tau, training):
        h = torch.tanh(self.in_proj(torch.cat([x, summary], -1)))
        logits = self.action_head(h) + self.param_to_action(p_lang)
        if float(self.stay_bias) != 0.0:
            # stay_bias is denominated in POST-temperature logit units: both gumbel_softmax and
            # the eval softmax divide logits by tau, so an unscaled bias b yields margin b/tau
            # (at tau=2, b=6 -> stay only ~0.90, not ~0.995). Multiplying by tau here makes the
            # realized stay mass ~= sigmoid-like in stay_bias regardless of the anneal point.
            bias = logits.new_zeros(logits.shape[-1]); bias[0] = float(self.stay_bias) * float(tau)
            logits = logits + bias
        if training:
            return F.gumbel_softmax(logits, tau=tau, hard=False)
        return F.softmax(logits / max(tau, 1e-6), -1)


# --------------------------------------------------------------------------- #
#  config + model
# --------------------------------------------------------------------------- #
class EMGSSMConfig(PretrainedConfig):
    model_type = "ssm_emg_ml"

    def __init__(self, vocab_size=32768, hidden_size=512, d_state=8, n_phase=6,
                 n_categories=16, k_params=8, dropout=0.1, gumbel_temperature=1.0,
                 lambda_cpc=0.1, lambda_success=0.05, lambda_cycle=0.0, param_reg=1e-2,
                 max_position_embeddings=1024, tie_word_embeddings=True, force_stay=False,
                 spine_read=True, param_film=True,
                 lambda_load=0.0, load_target_open=0.1, lambda_param_sup=0.0,
                 read_norm=True, a_log_min=math.log(0.02), dt_max=2.0,
                 logit_cap=6.0, param_label_smooth=0.05,
                 seal_read=True, seal_init=0.85,
                 bos_token_id=2, eos_token_id=3, pad_token_id=1, **kw):
        self.vocab_size = vocab_size; self.hidden_size = hidden_size
        self.d_state = d_state; self.n_phase = n_phase; self.n_categories = n_categories
        self.k_params = k_params; self.dropout = dropout; self.gumbel_temperature = gumbel_temperature
        self.lambda_cpc = lambda_cpc; self.lambda_success = lambda_success
        self.lambda_cycle = lambda_cycle; self.param_reg = param_reg
        self.max_position_embeddings = max_position_embeddings
        self.force_stay = force_stay
        self.spine_read = spine_read; self.param_film = param_film
        self.lambda_load = lambda_load; self.load_target_open = load_target_open
        self.lambda_param_sup = lambda_param_sup
        self.read_norm = read_norm            # (S2)
        self.a_log_min = a_log_min            # (S1)
        self.dt_max = dt_max                  # (S1)
        self.logit_cap = logit_cap            # (S3)
        self.param_label_smooth = param_label_smooth   # (S3)
        self.seal_read = seal_read            # (S4) graded parameter-driven read seal
        self.seal_init = seal_init            # (S4) initial seal (sigmoid space), 0.85 ~ near-uniform
        self.auto_map = {"AutoConfig": "emg_ssm_multilingual.EMGSSMConfig",
                         "AutoModelForCausalLM": "emg_ssm_multilingual.EMGSSMForCausalLM"}
        super().__init__(tie_word_embeddings=tie_word_embeddings, bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kw)


class EMGSSMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = EMGSSMConfig
    _tied_weights_keys = {"lm_head.weight": "wte.weight"}

    def __init__(self, cfg: EMGSSMConfig):
        super().__init__(cfg)
        H, K = cfg.hidden_size, cfg.n_categories
        self.wte = nn.Embedding(cfg.vocab_size, H)
        self.embed_drop = nn.Dropout(cfg.dropout)
        self.params = InputInferredParameters(H, cfg.k_params, cfg.gumbel_temperature, cfg.logit_cap)
        self.controller = PhaseController(H, cfg.k_params)
        self.ssm = PhaseSelectiveSSM(H, cfg.d_state, cfg.n_phase, cfg.k_params,
                                     cfg.param_film, cfg.spine_read, cfg.read_norm,
                                     cfg.a_log_min, cfg.dt_max,
                                     cfg.seal_read, cfg.seal_init)
        self.merge = nn.Linear(2 * H, H)
        self.cat_prior = nn.Linear(H, K)
        self.cat_emb = nn.Embedding(K, H)
        self.expect_proj = nn.Linear(H, H)
        self.realize_proj = nn.Linear(H, H)
        self.lm_head = nn.Linear(H, cfg.vocab_size, bias=False)
        self.K = K; self.pad_id = cfg.pad_token_id
        self._force_stay = False              # (S5-hard) runtime-only; not serialized
        self._aux = {}
        self.register_buffer("param_targets", torch.zeros(0), persistent=False)
        self.register_buffer("param_sup_mask", torch.zeros(0), persistent=False)
        self._lang_to_idx = {}
        self.post_init()

    def get_input_embeddings(self): return self.wte
    def set_input_embeddings(self, v): self.wte = v
    def get_output_embeddings(self): return self.lm_head
    def set_temperature(self, t): self.params.set_temperature(t); self.config.gumbel_temperature = float(t)
    def set_stay_bias(self, v): self.controller.set_stay_bias(v)   # (S5) soft phase curriculum

    def set_force_stay(self, v):
        """(S5-hard) True -> forward() bypasses the PhaseController entirely and uses a hard
        one-hot STAY action: depth is pinned at level 0 (no diffusion of the soft depth vector),
        the action head receives ZERO gradient (it cannot learn to cancel a bias, the failure
        mode observed with the soft warmup), and the model trains as a flat SSM. This is the
        true force_stay for the warmup curriculum; release with set_force_stay(False)."""
        self._force_stay = bool(v)

    @torch.no_grad()
    def infer_seal(self, input_ids, attention_mask=None):
        """Per-sequence read-seal scalar (B,1) in (0,1). Lower = harder island seal."""
        x = self.wte(input_ids)
        if attention_mask is None and self.pad_id is not None:
            attention_mask = (input_ids != self.pad_id).long()
        p = self.params(x, attention_mask)
        s = self.ssm.seal_from_params(p if self.config.param_film else None)
        return s if s is not None else x.new_zeros(input_ids.size(0), 1)

    def set_param_targets(self, langs, table=None, supervise=None, bit_assignment=None):
        table = table or ROBERTS_TABLE
        supervise = supervise or DEFAULT_SUPERVISE
        if bit_assignment is None:
            bit_assignment = {name: i for i, name in enumerate(supervise)}
        k = self.config.k_params
        self._lang_to_idx = {lg: i for i, lg in enumerate(langs)}
        tgt = torch.zeros(len(langs), k); mask = torch.zeros(k)
        for name in supervise:
            bit = bit_assignment[name]; mask[bit] = 1.0
            row = table[name]
            for li in range(len(langs)):
                tgt[li, bit] = float(row[li])
        dev = self.wte.weight.device
        self.param_targets = tgt.to(dev); self.param_sup_mask = mask.to(dev)
        return dict(self._lang_to_idx)

    @torch.no_grad()
    def infer_settings(self, input_ids, attention_mask=None):
        x = self.wte(input_ids)
        if attention_mask is None and self.pad_id is not None:
            attention_mask = (input_ids != self.pad_id).long()
        self.params(x, attention_mask)
        lg = self.params._last_logits
        return (torch.sigmoid(lg / float(self.params.temperature)) > 0.5).float()

    def _mos_logmarginal(self, summary, pi_log):
        B, H = summary.shape
        K = self.K
        comp = summary.unsqueeze(1) + self.cat_emb.weight.unsqueeze(0)     # (B,K,H)
        logits = self.lm_head(comp.reshape(B * K, H)).reshape(B, K, -1)    # (B,K,V)
        comp_logp = F.log_softmax(logits, dim=-1)
        return torch.logsumexp(pi_log.unsqueeze(-1) + comp_logp, dim=1)    # (B,V)

    def aux_loss(self):
        if not self._aux:
            return self.wte.weight.new_zeros(())
        c = self.config
        return (c.lambda_cpc * self._aux.get("cpc", 0.0)
                + c.lambda_success * self._aux.get("success", 0.0)
                + c.lambda_cycle * self._aux.get("cycle", 0.0)
                + c.param_reg * self._aux.get("param", 0.0)
                + getattr(c, "lambda_load", 0.0) * self._aux.get("load", 0.0)
                + getattr(c, "lambda_param_sup", 0.0) * self._aux.get("param_sup", 0.0))

    def aux_components(self):
        out = {}
        for k in ("cpc", "success", "cycle", "param", "load", "param_sup"):
            v = self._aux.get(k)
            if v is not None:
                out[k] = float(v.detach()) if torch.is_tensor(v) else float(v)
        return out

    def diagnostics(self):
        d = getattr(self, "_diag", {})
        return {k: (v.detach().item() if torch.is_tensor(v) else v) for k, v in d.items()}

    def forward(self, input_ids, attention_mask=None, labels=None, lang_ids=None, **kw):
        x = self.embed_drop(self.wte(input_ids))
        B, T, H = x.shape
        if attention_mask is None and self.pad_id is not None:
            attention_mask = (input_ids != self.pad_id).long()
        p = self.params(x, attention_mask)
        tau = float(self.params.temperature)
        self.ssm._film_p = p if self.config.param_film else None

        buf, depth = self.ssm.init_state(B, x.device, x.dtype)
        summary = x.new_zeros(B, H)
        prev_summary = summary
        logits, expect_reps, realize_reps, valid = [], [], [], []
        success_acc = x.new_zeros(())
        pri_entropy = x.new_zeros(()); post_entropy = x.new_zeros(()); n_pos = 0
        cat_ent_sum = x.new_zeros(()); pi_accum = x.new_zeros(self.K); valid_count = x.new_zeros(())
        act_sum = x.new_zeros(3); depth_acc = x.new_zeros(())
        seq_cat = x.new_zeros(B, self.K); seq_act = x.new_zeros(B, 3); seq_valid = x.new_zeros(B)  # per-seq probes
        level = torch.arange(self.ssm.P, device=x.device, dtype=x.dtype)
        m = attention_mask if attention_mask is not None else torch.ones(B, T, device=x.device)

        for t in range(T):
            xt = x[:, t]
            if getattr(self, "_force_stay", False):
                # (S5-hard) hard STAY: no controller in the graph, no gradient to the action
                # head, depth stays exactly one-hot at its current level (no random-walk mixing).
                action = x.new_zeros(B, 3); action[:, 0] = 1.0
            else:
                action = self.controller(xt, summary, p, tau, self.training)
            if self.config.force_stay:
                action = torch.zeros_like(action); action[:, 0] = 1.0
            buf, depth, read, success = self.ssm.step(buf, depth, xt, action)
            success_acc = success_acc + success
            prev_summary = summary
            summary = torch.tanh(self.merge(torch.cat([summary, read], -1)))

            pi_log = F.log_softmax(self.cat_prior(summary), -1)
            logits.append(self._mos_logmarginal(summary, pi_log))

            with torch.no_grad():
                pim = pi_log.exp(); msk_t = m[:, t]
                cat_ent_sum = cat_ent_sum - (pim * pi_log).sum(-1).mul(msk_t).sum()
                pi_accum = pi_accum + (pim * msk_t.unsqueeze(-1)).sum(0)
                valid_count = valid_count + msk_t.sum()
                act_sum = act_sum + (action * msk_t.unsqueeze(-1)).sum(0)
                depth_acc = depth_acc + ((depth * level).sum(-1) * msk_t).sum()
                seq_cat = seq_cat + pim * msk_t.unsqueeze(-1)         # per-seq category posterior
                seq_act = seq_act + action * msk_t.unsqueeze(-1)      # per-seq action mix
                seq_valid = seq_valid + msk_t

            expect_reps.append(self.expect_proj(prev_summary))
            realize_reps.append(self.realize_proj(xt))
            valid.append(m[:, t])

            if self.config.lambda_cycle > 0 and labels is not None:
                w = input_ids[:, t]
                comp = summary.unsqueeze(1) + self.cat_emb.weight.unsqueeze(0)
                comp_logits = self.lm_head(comp.reshape(B * self.K, H)).reshape(B, self.K, -1)
                comp_lp_w = F.log_softmax(comp_logits, -1).gather(
                    2, w[:, None, None].expand(B, self.K, 1)).squeeze(-1)
                post = F.softmax(pi_log + comp_lp_w, -1)
                msk = m[:, t]
                post_entropy = post_entropy - (post * post.clamp_min(1e-9).log()).sum(-1).mul(msk).sum()
                pri_entropy = pri_entropy - (pi_log.exp() * pi_log).sum(-1).mul(msk).sum()
                n_pos += int(msk.sum().item())

        logits = torch.stack(logits, 1)

        g = torch.stack(expect_reps, 1).reshape(B * T, H)
        f = torch.stack(realize_reps, 1).reshape(B * T, H)
        v = torch.stack(valid, 1).reshape(B * T) > 0.5
        if v.any():
            g = F.normalize(g[v], dim=-1); f = F.normalize(f[v], dim=-1)
            sim = g @ f.t() / 0.1
            cpc = F.cross_entropy(sim, torch.arange(sim.size(0), device=sim.device))
        else:
            cpc = x.new_zeros(())
        self._aux = {"cpc": cpc, "success": success_acc / max(1, T),
                     "param": self.params.ig_penalty()}
        if self.config.lambda_cycle > 0 and labels is not None and n_pos > 0:
            self._aux["cycle"] = (post_entropy - pri_entropy) / n_pos

        vc = valid_count.clamp(min=1.0)
        pi_mean = pi_accum / vc
        mean_act = act_sum / vc
        self._diag = {"cat_entropy": (cat_ent_sum / vc).detach(),
                      "n_active": (pi_mean > (0.5 / self.K)).sum().detach(),
                      "open_rate": mean_act[1].detach(),
                      "close_rate": mean_act[2].detach(),
                      "mean_depth": (depth_acc / vc).detach()}

        # per-sequence features for probe_parameters_ssmemg.py
        sv = seq_valid.clamp(min=1.0)
        self._seq_cat = (seq_cat / sv.unsqueeze(-1)).detach()        # (B,K) category posterior
        self._seq_act = (seq_act / sv.unsqueeze(-1)).detach()        # (B,3) realized action mix

        if getattr(self.config, "lambda_load", 0.0) > 0:
            r = float(self.config.load_target_open)
            target = mean_act.new_tensor([max(1e-3, 1 - 2 * r), r, r])
            ma = mean_act.clamp_min(1e-6)
            self._aux["load"] = (ma * (ma / target).log()).sum()

        if (getattr(self.config, "lambda_param_sup", 0.0) > 0 and lang_ids is not None
                and self.param_targets.numel() > 0):
            s = float(self.config.param_label_smooth)
            tgt = self.param_targets[lang_ids] * (1 - 2 * s) + s   # (S3) {0,1} -> {s,1-s}
            logit = self.params._last_logits
            bce = F.binary_cross_entropy_with_logits(logit, tgt, reduction="none")
            msk = self.param_sup_mask.unsqueeze(0)
            denom = (msk.sum() * B).clamp(min=1.0)
            self._aux["param_sup"] = (bce * msk).sum() / denom

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   labels.reshape(-1), ignore_index=-100) + self.aux_loss()
        return CausalLMOutputWithPast(loss=loss, logits=logits)


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = EMGSSMConfig(vocab_size=200, hidden_size=48, d_state=4, n_phase=4, n_categories=8,
                       k_params=8, lambda_cycle=0.1, lambda_load=5e-3, lambda_param_sup=0.1,
                       spine_read=True, param_film=True, read_norm=True, logit_cap=6.0)
    model = EMGSSMForCausalLM(cfg)
    lang_map = model.set_param_targets(["eng", "nld", "zho"], supervise=["v2", "filler_gap"])
    ids = torch.randint(0, 200, (3, 16))
    lang_ids = torch.tensor([lang_map[l] for l in ("eng", "nld", "zho")])

    # Stress test: feed LARGE-magnitude embeddings to confirm the read/state stay bounded
    # (this is what the old unbounded model blew up on).
    with torch.no_grad():
        model.wte.weight.mul_(8.0)
    out = model(input_ids=ids, labels=ids, lang_ids=lang_ids)
    comp = model.aux_components(); diag = model.diagnostics()
    print("logits", tuple(out.logits.shape), "| loss", round(float(out.loss.detach()), 4),
          "| finite", bool(torch.isfinite(out.loss).item()))
    print("marginal sums to 1?", round(float(out.logits[0, 0].exp().sum()), 5))
    print("aux:", {k: round(v, 4) for k, v in comp.items()})
    print("diag:", {k: round(v, 4) for k, v in diag.items()})
    print("router logit |max|:", round(float(model.params._last_logits.abs().max()), 3),
          "(cap", cfg.logit_cap, ")")

    # gradient magnitude check on the previously-exploding params
    out.loss.backward()
    def gn(p): return round(float(p.grad.norm()), 3) if p.grad is not None else None
    print("grad norms:", {"merge": gn(model.merge.weight), "read_out": gn(model.ssm.read_out.weight),
                          "dt_proj": gn(model.ssm.dt_proj.weight),
                          "action_head": gn(model.controller.action_head.weight),
                          "router": gn(model.params.router[-1].weight),
                          "seal_from_p": gn(model.ssm.seal_from_p.weight),
                          "seal_bias": gn(model.ssm.seal_bias)})

    # (S4) seal is per-sequence, p-driven, and in (0,1)
    seal = model.infer_seal(ids)
    print("seal (per-seq):", [round(float(s), 3) for s in seal.squeeze(-1)],
          "| grad on seal path:", model.ssm.seal_from_p.weight.grad is not None)

    # (S5) large stay_bias -> STAY dominates the action mix (curriculum warmup works)
    model.set_stay_bias(8.0)
    model.eval()
    with torch.no_grad():
        _ = model(input_ids=ids, labels=ids, lang_ids=lang_ids)
    print("stay_bias=8 -> open_rate {:.3f} close_rate {:.3f} (both should be ~0; bias is tau-scaled)".format(
        float(model.diagnostics()["open_rate"]), float(model.diagnostics()["close_rate"])))
    model.set_stay_bias(0.0)

    # (S5-hard) force_stay -> EXACTLY zero opens/closes, depth pinned, no controller gradient
    model.set_force_stay(True)
    model.zero_grad(set_to_none=True)          # isolate: clear grads from the earlier backward
    model.train()
    out_fs = model(input_ids=ids, labels=ids, lang_ids=lang_ids)
    (out_fs.loss + model.aux_loss()).backward()
    ah_grad = model.controller.action_head.weight.grad
    ah_norm = float(ah_grad.norm()) if ah_grad is not None else 0.0
    d_fs = model.diagnostics()
    print("force_stay=True -> open {:.4f} close {:.4f} depth {:.4f} action_head_grad {:.2e} "
          "(all must be exactly 0)".format(float(d_fs["open_rate"]), float(d_fs["close_rate"]),
                                           float(d_fs["mean_depth"]), ah_norm))
    model.set_force_stay(False)
    model.zero_grad(set_to_none=True)