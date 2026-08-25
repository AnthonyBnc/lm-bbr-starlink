import math
import pickle
import unittest

from plm_special.utils.constants import ACTION_LEVELS as LEGACY_ACTION_LEVELS
from utils.bbr import (
    ACTION_LEVELS,
    BBR_PHASES,
    BW_CRUISE,
    BW_DOWN,
    BW_UP,
    PACING_GAINS,
    action_index_to_gain,
    mask_action_values,
    phase_action_mask,
    valid_action_indices,
    valid_pacing_gains,
    validate_action_index,
    validate_phase_action,
)
from utils.exp_pool import ExperiencePool


class BBRActionContractTests(unittest.TestCase):
    def test_action_mapping_is_the_locked_eleven_gain_contract(self):
        self.assertEqual(ACTION_LEVELS, 11)
        self.assertEqual(LEGACY_ACTION_LEVELS, ACTION_LEVELS)
        self.assertEqual(
            PACING_GAINS,
            (0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25),
        )
        self.assertEqual(
            tuple(action_index_to_gain(index) for index in range(ACTION_LEVELS)),
            PACING_GAINS,
        )

    def test_phase_action_indices_are_fixed(self):
        self.assertEqual(BBR_PHASES, (BW_DOWN, BW_CRUISE, BW_UP))
        self.assertEqual(valid_action_indices(BW_DOWN), (0, 1, 2, 3, 4))
        self.assertEqual(valid_action_indices(BW_CRUISE), (5,))
        self.assertEqual(valid_action_indices(BW_UP), (6, 7, 8, 9, 10))

    def test_phase_masks_allow_only_documented_gains(self):
        expected = {
            BW_DOWN: (0.90, 0.92, 0.94, 0.96, 0.98),
            BW_CRUISE: (1.00,),
            BW_UP: (1.05, 1.10, 1.15, 1.20, 1.25),
        }

        for phase, expected_gains in expected.items():
            mask = phase_action_mask(phase)
            self.assertEqual(len(mask), ACTION_LEVELS)
            self.assertEqual(valid_pacing_gains(phase), expected_gains)
            self.assertEqual(
                tuple(gain for gain, is_valid in zip(PACING_GAINS, mask) if is_valid),
                expected_gains,
            )

    def test_invalid_action_values_are_masked_before_selection(self):
        values = tuple(float(index) for index in range(ACTION_LEVELS))
        masked = mask_action_values(values, BW_CRUISE)

        self.assertEqual(masked[5], values[5])
        invalid_values = (value for i, value in enumerate(masked) if i != 5)
        self.assertTrue(all(math.isinf(value) and value < 0 for value in invalid_values))

    def test_invalid_contract_inputs_fail_explicitly(self):
        with self.assertRaises(ValueError):
            valid_action_indices("STARTUP")
        with self.assertRaises(ValueError):
            validate_action_index(11)
        with self.assertRaises(TypeError):
            validate_action_index(1.0)
        with self.assertRaises(ValueError):
            mask_action_values([0.0] * 12, BW_DOWN)

    def test_phase_action_pairs_are_validated(self):
        self.assertEqual(validate_phase_action(BW_DOWN, 4), 4)
        self.assertEqual(validate_phase_action(BW_CRUISE, 5), 5)
        self.assertEqual(validate_phase_action(BW_UP, 6), 6)

        with self.assertRaises(ValueError):
            validate_phase_action(BW_DOWN, 10)

    def test_experience_pool_persists_phase_metadata(self):
        pool = ExperiencePool()
        pool.add([1, 2, 3], 5, 1.0, False, BW_CRUISE)

        restored = pickle.loads(pickle.dumps(pool))
        self.assertEqual(restored.phases, [BW_CRUISE])
        self.assertEqual(restored.actions, [5])

        with self.assertRaises(ValueError):
            pool.add([1, 2, 3], 0, 1.0, False, BW_UP)


if __name__ == "__main__":
    unittest.main()
