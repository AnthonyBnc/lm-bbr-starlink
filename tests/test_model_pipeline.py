from types import SimpleNamespace
import unittest

import torch
from torch import nn

from plm_special.models.rl_policy import OfflineRLPolicy
from plm_special.models.state_encoder import EncoderNetwork
from utils.bbr import ACTION_LEVELS


class DummyBackbone(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(max_position_embeddings=64)

    def forward(self, inputs_embeds, attention_mask=None, output_hidden_states=False):
        return SimpleNamespace(
            last_hidden_state=inputs_embeds * self.anchor,
            hidden_states=(inputs_embeds * self.anchor,),
        )


class ModelPipelineTests(unittest.TestCase):
    def test_ninth_state_feature_uses_fc9(self):
        encoder = EncoderNetwork(embed_dim=1)
        with torch.no_grad():
            encoder.fc8[0].weight.zero_()
            encoder.fc8[0].bias.zero_()
            encoder.fc9[0].weight.zero_()
            encoder.fc9[0].bias.fill_(1.0)
        outputs = encoder(torch.zeros(1, 1, 9))
        self.assertEqual(float(outputs[7].item()), 0.0)
        self.assertEqual(float(outputs[8].item()), 1.0)

    def test_policy_forward_returns_eleven_logits_per_position(self):
        encoder = EncoderNetwork(embed_dim=4)
        policy = OfflineRLPolicy(
            state_feature_dim=4,
            action_levels=ACTION_LEVELS,
            state_encoder=encoder,
            plm=DummyBackbone(),
            plm_embed_size=8,
            max_length=2,
            max_ep_len=10,
            device="cpu",
        )
        logits = policy(
            states=torch.zeros(1, 2, 9),
            actions=torch.zeros(1, 2, 1),
            returns=torch.zeros(1, 2, 1),
            timesteps=torch.tensor([[0, 1]]),
        )
        self.assertEqual(tuple(logits.shape), (1, 2, 11))

    def test_quantum_policy_forward_returns_eleven_logits_per_position(self):
        encoder = EncoderNetwork(embed_dim=4)
        policy = OfflineRLPolicy(
            state_feature_dim=4,
            action_levels=ACTION_LEVELS,
            state_encoder=encoder,
            plm=DummyBackbone(),
            plm_embed_size=8,
            max_length=2,
            max_ep_len=10,
            device="cpu",
            head_type="quantum",
            quantum_config={"n_qubits": 4, "depth": 2},
        )
        logits = policy(
            states=torch.zeros(1, 2, 9),
            actions=torch.zeros(1, 2, 1),
            returns=torch.zeros(1, 2, 1),
            timesteps=torch.tensor([[0, 1]]),
        )
        self.assertEqual(tuple(logits.shape), (1, 2, 11))
        self.assertEqual(policy.quantum_config["n_qubits"], 4)


if __name__ == "__main__":
    unittest.main()
