"""Classical task heads used as controls for Quantum-GPT."""

from torch import nn

from utils.bbr import ACTION_LEVELS


class ClassicalBottleneckActionHead(nn.Module):
    """Parameter-light classical twin for the quantum bottleneck path."""

    def __init__(
        self,
        input_dim,
        action_levels=ACTION_LEVELS,
        bottleneck_dim=4,
        input_layernorm=False,
        temperature=1.0,
    ):
        super().__init__()
        if action_levels != ACTION_LEVELS:
            raise ValueError(
                "ClassicalBottleneckActionHead requires {} actions".format(ACTION_LEVELS)
            )
        if bottleneck_dim < 1:
            raise ValueError("bottleneck_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.input_dim = input_dim
        self.action_levels = action_levels
        self.bottleneck_dim = bottleneck_dim
        self.input_layernorm = bool(input_layernorm)
        self.temperature = float(temperature)
        self.input_norm = nn.LayerNorm(input_dim) if self.input_layernorm else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.Tanh(),
            nn.Linear(bottleneck_dim, action_levels),
        )
        self.diagnostics_enabled = False
        self.last_diagnostics = None

    def forward(self, hidden_states):
        projected = self.net[0](self.input_norm(hidden_states))
        bottleneck = self.net[1](projected / self.temperature)
        logits = self.net[2](bottleneck)
        if self.diagnostics_enabled:
            self.last_diagnostics = {
                "projected_abs_mean": projected.detach().float().abs().mean().item(),
                "bottleneck_std": bottleneck.detach().float().std(unbiased=False).item(),
                "bottleneck_saturation_ratio": (
                    bottleneck.detach().float().abs() > 0.99
                ).float().mean().item(),
            }
        return logits

    def enable_diagnostics(self, enabled=True):
        self.diagnostics_enabled = bool(enabled)
        self.last_diagnostics = None

    def manifest_config(self):
        return {
            "type": "classical_bottleneck",
            "bottleneck_dim": self.bottleneck_dim,
            "activation": "tanh",
            "input_layernorm": self.input_layernorm,
            "temperature": self.temperature,
        }
