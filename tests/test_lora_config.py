import unittest

from config import cfg


class ModernLoraConfigTests(unittest.TestCase):
    UNDER_400M_MODEL_KEYS = {
        "granite_4_0_350m",
        "pleias_rag_350m",
        "lfm2_5_350m",
        "granite_4_0_h_350m",
        "gemma_3_270m",
    }

    def test_all_modern_models_have_lora_configs(self):
        self.assertEqual(
            set(cfg.modern_model_registry),
            set(cfg.modern_lora_registry),
        )

    def test_under_400m_models_are_pinned_and_pending_smoke(self):
        self.assertTrue(
            self.UNDER_400M_MODEL_KEYS.issubset(cfg.modern_model_registry)
        )
        for model_key in self.UNDER_400M_MODEL_KEYS:
            with self.subTest(model_key=model_key):
                model = cfg.modern_model_registry[model_key]
                self.assertEqual(len(model["revision"]), 40)
                self.assertEqual(model["loader_status"], "registered_pending_smoke")

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
