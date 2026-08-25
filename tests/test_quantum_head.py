import unittest

import torch

from plm_special.quantum_head import QuantumActionHead
from utils.bbr import ACTION_LEVELS


class QuantumHeadTests(unittest.TestCase):
    def test_quantum_head_outputs_eleven_logits_and_gradients(self):
        head = QuantumActionHead(input_dim=8, n_qubits=4, depth=2)
        hidden = torch.randn(2, 3, 8, requires_grad=True)
        logits = head(hidden)
        self.assertEqual(tuple(logits.shape), (2, 3, ACTION_LEVELS))
        loss = logits.square().mean()
        loss.backward()
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertTrue(torch.isfinite(head.quantum_layer.weight.grad).all())

    def test_quantum_head_manifest_records_default_vqc(self):
        head = QuantumActionHead(input_dim=8, n_qubits=4, depth=2)
        config = head.manifest_config()
        self.assertEqual(config["n_qubits"], 4)
        self.assertEqual(config["depth"], 2)
        self.assertEqual(config["ansatz"], "trainable_ry_layers")
        self.assertEqual(config["trainable_circuit_parameters"], 8)
        self.assertEqual(config["framework"], "qiskit")
        self.assertEqual(config["measurement"], "per_qubit_pauli_z_expectation")
        self.assertEqual(config["estimator"], "qiskit.primitives.StatevectorEstimator")
        self.assertEqual(config["default_precision"], 0.0)
        self.assertIsNone(config["shots"])

    def test_quantum_head_supports_ry_rz_ansatz(self):
        head = QuantumActionHead(
            input_dim=8,
            n_qubits=4,
            depth=2,
            ansatz="trainable_ry_rz_layers",
        )
        hidden = torch.randn(1, 2, 8, requires_grad=True)
        logits = head(hidden)
        self.assertEqual(tuple(logits.shape), (1, 2, ACTION_LEVELS))
        logits.sum().backward()
        self.assertIsNotNone(hidden.grad)
        config = head.manifest_config()
        self.assertEqual(config["ansatz"], "trainable_ry_rz_layers")
        self.assertEqual(config["trainable_circuit_parameters"], 16)

    def test_quantum_head_records_trainability_diagnostics(self):
        head = QuantumActionHead(input_dim=8, n_qubits=4, depth=2)
        head.enable_diagnostics()
        head(torch.randn(1, 2, 8))
        self.assertIn("bottleneck_std", head.last_diagnostics)
        self.assertIn("bottleneck_saturation_ratio", head.last_diagnostics)
        self.assertGreaterEqual(head.last_diagnostics["bottleneck_saturation_ratio"], 0.0)
        self.assertLessEqual(head.last_diagnostics["bottleneck_saturation_ratio"], 1.0)

    def test_quantum_head_supports_normalized_temperature_scaled_encoding(self):
        head = QuantumActionHead(
            input_dim=8,
            n_qubits=4,
            depth=2,
            input_layernorm=True,
            temperature=4.0,
            angle_scale="half_pi",
        )
        head.enable_diagnostics()
        logits = head(torch.randn(1, 2, 8))
        self.assertEqual(tuple(logits.shape), (1, 2, ACTION_LEVELS))
        self.assertIn("raw_projection_abs_mean", head.last_diagnostics)
        self.assertIn("angle_abs_mean", head.last_diagnostics)
        config = head.manifest_config()
        self.assertTrue(config["input_layernorm"])
        self.assertEqual(config["temperature"], 4.0)
        self.assertEqual(config["angle_scale"], "half_pi")


if __name__ == "__main__":
    unittest.main()
