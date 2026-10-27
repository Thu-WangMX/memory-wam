"""Learning-rate multipliers for warmup, a stable plateau, and cosine decay."""
import math


class WarmupStableDecay:
    """Callable stored by LambdaLR; step counts refer to optimizer updates."""

    def __init__(self, total_steps, warmup_steps, decay_steps, min_ratio=0.01):
        for name, value in [('total_steps', total_steps), ('warmup_steps', warmup_steps),
                            ('decay_steps', decay_steps)]:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f'{name} must be an integer')
        if total_steps <= 0 or warmup_steps < 0 or decay_steps <= 0:
            raise ValueError('WSD requires total > 0, warmup >= 0, decay > 0')
        if warmup_steps + decay_steps > total_steps:
            raise ValueError('WSD warmup + decay exceeds total training steps')
        if not math.isfinite(min_ratio) or not 0 <= min_ratio <= 1:
            raise ValueError('WSD min_ratio must be finite and in [0, 1]')
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_start = total_steps - decay_steps
        self.min_ratio = float(min_ratio)

    def __call__(self, step):
        if step < self.warmup_steps:
            return min(1.0, (step + 1) / self.warmup_steps)
        if step <= self.decay_start:
            return 1.0
        progress = min(1.0, (step - self.decay_start) / self.decay_steps)
        return self.min_ratio + (1.0 - self.min_ratio) * (1.0 + math.cos(math.pi * progress)) / 2
