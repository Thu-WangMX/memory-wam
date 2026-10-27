"""Third MoT experts: deterministic queries or ordered-block flow inputs."""
import torch
from torch import nn

from .action_dit import ActionDiT
from .mot import MoT
from .wan_video_dit import sinusoidal_embedding_1d


class PastActionExpert(ActionDiT):
    FRESH_PREFIXES = ('query', 'phase_embedding.', 'relative_time_embedding.', 'head.')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # No input projection for ground-truth or noised DreamDojo labels.
        del self.action_encoder
        self.query = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)
        self.phase_embedding = nn.Embedding(4, self.hidden_dim, padding_idx=0)
        self.relative_time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim))

    def pre_queries(self, phase, relative_time, context, context_mask, freqs):
        if (phase.ndim != 2 or relative_time.shape != phase.shape
                or phase.shape[0] != context.shape[0]
                or phase.min() < 0 or phase.max() > 3
                or not torch.isfinite(relative_time).all()
                or (not getattr(self, 'allow_future', False) and (relative_time > 0).any())):
            raise ValueError('Invalid causal past-action queries')
        b, slots = phase.shape
        rel = sinusoidal_embedding_1d(self.freq_dim, relative_time.reshape(-1))
        rel = self.relative_time_embedding(rel.to(self.query.dtype)).reshape(b, slots, -1)
        tokens = self.query + self.phase_embedding(phase) + rel
        zero_time = torch.zeros(b, device=tokens.device, dtype=tokens.dtype)
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, zero_time))
        return dict(tokens=tokens, freqs=freqs, t=t,
                    t_mod=self.time_projection(t).unflatten(1, (6, self.hidden_dim)),
                    context=self.text_embedding(context),
                    context_mask=context_mask[:, None, :].expand(-1, slots, -1))

    def initialize_backbone(self, action_expert):
        source = {k: v for k, v in action_expert.state_dict().items()
                  if not k.startswith(('action_encoder.', 'head.'))}
        result = self.load_state_dict(source, strict=False)
        expected = {k for k in self.state_dict()
                    if k.startswith(self.FRESH_PREFIXES)}
        if set(result.missing_keys) != expected or result.unexpected_keys:
            raise ValueError(f'Unexpected past-action backbone transfer: {result}')


class PastActionIntervalExpert(PastActionExpert):
    FRESH_PREFIXES = PastActionExpert.FRESH_PREFIXES + ('interval_length_embedding.',)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The endpoint age plus this observed length defines the entire interval.
        self.interval_length_embedding = nn.Embedding(17, self.hidden_dim, padding_idx=0)


class PastActionIntervalFlowExpert(PastActionIntervalExpert):
    FRESH_PREFIXES = PastActionIntervalExpert.FRESH_PREFIXES + ('noisy_latent_embedding.',)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.noisy_latent_embedding = nn.Linear(self.action_dim, self.hidden_dim)

    def add_flow_inputs(self, pre, noisy, timestep):
        pre = dict(pre)
        pre['tokens'] = pre['tokens'] + self.noisy_latent_embedding(
            noisy.flatten(-2).to(self.query.dtype))
        t = self.time_embedding(sinusoidal_embedding_1d(
            self.freq_dim, timestep.to(self.query.dtype)))
        pre.update(t=t, t_mod=self.time_projection(t).unflatten(1, (6, self.hidden_dim)))
        return pre


class PastActionMoT(MoT):
    def prefill_past_action(self, pre, video_kv, valid):
        """Compute third-expert states once, append layer-aligned K/V for actions.

        Every cached K/V is taken BEFORE that layer's update, exactly as in the
        training joint MoT. The past expert cannot see future video or actions.
        """
        expert = self.mixtures['past_action']
        x = pre['tokens']
        if valid.shape != x.shape[:2] or len(video_kv) != self.num_layers:
            raise ValueError('Past-action cache shape/layer mismatch')
        b, slots = valid.shape
        h = video_kv[0]['k'].shape[1]
        history_valid = pre.get('history_key_valid')
        if history_valid is None:
            history_valid = torch.ones(b, h, device=x.device, dtype=torch.bool)
        if history_valid.shape != (b, h):
            raise ValueError('Past-action history validity shape mismatch')
        key_valid = torch.cat((history_valid.to(device=x.device, dtype=torch.bool), valid), 1)
        mask = key_valid[:, None, :].expand(-1, slots, -1).clone()
        if 'query_attention_mask' in pre:
            relation = pre['query_attention_mask'].to(device=x.device, dtype=torch.bool)
            if relation.shape != (slots, slots):
                raise ValueError('Invalid transition-query attention mask')
            mask[:, :, h:] &= relation
        cache = []
        for index in range(self.num_layers):
            block = expert.blocks[index]
            q, k, v, residual, gate, shift, scale, mlp_gate, checkpoint = self._build_expert_attention_io(
                expert, block, x, pre['freqs'], pre['t_mod'])
            k = torch.cat((video_kv[index]['k'], k), 1)
            v = torch.cat((video_kv[index]['v'], v), 1)
            cache.append(dict(k=k, v=v))
            mixed = self._mixed_attention(q, k, v, mask)
            x = self._apply_post_with_optional_checkpoint(
                block, residual, gate, shift, scale, mlp_gate, checkpoint, mixed,
                dict(context=pre['context'], mask=pre['context_mask']))
        return cache, key_valid, x
