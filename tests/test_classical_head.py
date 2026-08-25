import unittest

import torch

from plm_special.classical_head import ClassicalBottleneckActionHead


class ClassicalHeadTests(unittest.TestCase):
    def test_bottleneck_head_records_trainability_diagnostics(self):
        head = ClassicalBottleneckActionHead(input_dim=8, bottleneck_dim=4)
        head.enable_diagnostics()
        logits = head(torch.randn(1, 3, 8))
        self.assertEqual(tuple(logits.shape), (1, 3, 11))
        self.assertIn("bottleneck_std", head.last_diagnostics)
        self.assertIn("bottleneck_saturation_ratio", head.last_diagnostics)
        self.assertGreaterEqual(head.last_diagnostics["bottleneck_saturation_ratio"], 0.0)
        self.assertLessEqual(head.last_diagnostics["bottleneck_saturation_ratio"], 1.0)

    def test_bottleneck_head_records_normalization_and_temperature(self):
        head = ClassicalBottleneckActionHead(
            input_dim=8,
            bottleneck_dim=4,
            input_layernorm=True,
            temperature=4.0,
        )
        config = head.manifest_config()
        self.assertTrue(config["input_layernorm"])
        self.assertEqual(config["temperature"], 4.0)


if __name__ == "__main__":
    unittest.main()
