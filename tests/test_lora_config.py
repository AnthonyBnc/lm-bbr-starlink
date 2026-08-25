import unittest

from config import cfg


class ModernLoraConfigTests(unittest.TestCase):
    def test_all_modern_models_have_lora_configs(self):
        self.assertEqual(
            set(cfg.modern_model_registry),
            set(cfg.modern_lora_registry),
        )

    def test_lora_defaults_are_explicit_and_exploratory(self):
        self.assertGreater(cfg.lora_defaults["construction_rank"], 0)
        self.assertGreater(cfg.lora_defaults["alpha"], 0)
        self.assertGreaterEqual(cfg.lora_defaults["dropout"], 0.0)
        self.assertIn("exploratory", cfg.lora_defaults["status"])

    def test_every_family_declares_expected_adapter_count(self):
        for model_key, family in cfg.modern_lora_registry.items():
            with self.subTest(model_key=model_key):
                self.assertTrue(family["target_modules"])
                self.assertGreater(family["expected_adapter_modules"], 0)


if __name__ == "__main__":
    unittest.main()
