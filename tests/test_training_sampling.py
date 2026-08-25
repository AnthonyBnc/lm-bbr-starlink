from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from utils.bbr import BW_CRUISE, BW_DOWN, BW_UP
from utils.training_sampling import (
    TRAINING_SAMPLING_CRUISE80,
    TRAINING_SAMPLING_UP_FOCUS,
    build_training_window_indices,
    load_training_window_plan,
)


class SyntheticWindowDataset:
    def __init__(self):
        self.windows = [
            ([5, 5], [BW_CRUISE, BW_CRUISE]),
            ([0, 5], [BW_DOWN, BW_CRUISE]),
            ([6, 6], [BW_UP, BW_UP]),
            ([7, 5], [BW_UP, BW_CRUISE]),
            ([8, 8], [BW_UP, BW_UP]),
            ([9, 5], [BW_UP, BW_CRUISE]),
            ([10, 10], [BW_UP, BW_UP]),
        ]
        self.window_locations = ["London"] * len(self.windows)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        actions, phases = self.windows[index]
        return [], actions, [], [], phases


class TrainingSamplingTests(unittest.TestCase):
    def test_up_focused_sampling_is_deterministic_and_exact_length(self):
        dataset = SyntheticWindowDataset()
        first, first_record = build_training_window_indices(
            dataset,
            strategy=TRAINING_SAMPLING_UP_FOCUS,
            target_windows=1500,
            seed=100003,
        )
        second, second_record = build_training_window_indices(
            dataset,
            strategy=TRAINING_SAMPLING_UP_FOCUS,
            target_windows=1500,
            seed=100003,
        )
        self.assertEqual(first, second)
        self.assertEqual(first_record, second_record)
        self.assertEqual(len(first), 1500)
        self.assertGreater(first_record["selected_distribution"]["repeated_windows"], 0)

    def test_up_focused_sampling_increases_up_share(self):
        dataset = SyntheticWindowDataset()
        selected, record = build_training_window_indices(
            dataset,
            strategy=TRAINING_SAMPLING_UP_FOCUS,
            target_windows=1500,
            seed=100003,
            up_density_power=3.0,
            down_penalty=1.0,
        )
        selected_phases = Counter(
            phase for index in selected for phase in dataset[index][-1]
        )
        self.assertGreater(selected_phases[BW_UP] / (len(selected) * 2), 0.65)
        self.assertEqual(
            record["selected_distribution"]["phase_positions"][BW_UP],
            selected_phases[BW_UP],
        )
        up_actions = record["selected_distribution"]["action_positions"]
        self.assertTrue(all(up_actions[str(action)] > 0 for action in range(6, 11)))

    def test_original_strategy_rejects_silent_reduction(self):
        with self.assertRaisesRegex(ValueError, "original strategy"):
            build_training_window_indices(SyntheticWindowDataset(), target_windows=3)

    def test_precomputed_plan_reloads_exact_indices(self):
        dataset = SyntheticWindowDataset()
        indices, record = build_training_window_indices(
            dataset,
            strategy=TRAINING_SAMPLING_UP_FOCUS,
            target_windows=25,
            seed=100003,
            up_density_power=3.0,
            down_penalty=1.0,
        )
        plan = {
            "training_sampling": {
                key: record[key]
                for key in (
                    "strategy",
                    "seed",
                    "target_windows",
                    "with_replacement",
                    "up_density_power",
                    "down_penalty",
                    "up_action_window_quotas",
                )
            },
            "source_distribution": record["source_distribution"],
            "selected_distribution": record["selected_distribution"],
            "source_location_windows": record["source_location_windows"],
            "selected_location_windows": record["selected_location_windows"],
            "selected_dataset_indices": indices,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            loaded, loaded_record = load_training_window_plan(
                path,
                dataset,
                TRAINING_SAMPLING_UP_FOCUS,
                25,
                100003,
                3.0,
                1.0,
            )
        self.assertEqual(loaded, indices)
        self.assertEqual(
            loaded_record["selected_distribution"],
            record["selected_distribution"],
        )

    def test_cruise80_strategy_preserves_budget_locations_and_up_actions(self):
        dataset = SyntheticWindowDataset()
        dataset.windows = []
        dataset.window_locations = []
        for location in ("London", "Mumbai", "Ohio", "SaoPaulo", "Sydney"):
            for _ in range(5):
                dataset.windows.append(([5, 5], [BW_CRUISE, BW_CRUISE]))
                dataset.window_locations.append(location)
            for action in range(6, 11):
                dataset.windows.append(([action, action], [BW_UP, BW_UP]))
                dataset.window_locations.append(location)
        selected, record = build_training_window_indices(
            dataset,
            strategy=TRAINING_SAMPLING_CRUISE80,
            target_windows=50,
            replacement_windows=25,
            seed=100003,
            up_density_power=3.0,
            down_penalty=1.0,
        )
        self.assertEqual(len(selected), 50)
        self.assertEqual(
            record["selected_location_windows"],
            {location: 10 for location in dataset.window_locations[::10]},
        )
        actions = record["selected_distribution"]["action_positions"]
        self.assertEqual(len({actions[str(action)] for action in range(6, 11)}), 1)


if __name__ == "__main__":
    unittest.main()
