import unittest

import numpy as np

from plm_special.data.dataset import episode_window_starts
from utils.bbr import BW_CRUISE
from utils.exp_pool import ExperiencePool
from utils.splits import stratified_trace_split
from utils.starlink_preprocessing import DEVELOPMENT_LOCATIONS, STREAM_FLAGS


class SplitAndSequenceTests(unittest.TestCase):
    def _full_synthetic_pool(self):
        pool = ExperiencePool()
        source_files = []
        for stream_index, stream in enumerate(STREAM_FLAGS):
            for location_index, location in enumerate(DEVELOPMENT_LOCATIONS):
                for run in range(10):
                    path = "{}/{}/bbr_{}__REV_run{}.json".format(
                        stream, location, location, run + 1
                    )
                    source_files.append({"path": path, "intervals": 3, "sha256": path})
                    for interval in range(3):
                        pool.add(
                            state=np.asarray(
                                [location_index, stream_index, interval, 1, 0, 1, 1, 1, 1],
                                dtype=np.float32,
                            ),
                            action=5,
                            reward=float(interval),
                            done=interval == 2,
                            phase=BW_CRUISE,
                            sample_id="{}:{}".format(path, interval),
                        )
        pool.metadata = {
            "dataset_version": "synthetic",
            "held_out_location": "Tokyo",
            "source_files": source_files,
        }
        return pool

    def test_trace_split_is_deterministic_and_disjoint(self):
        pool = self._full_synthetic_pool()
        train_a, validation_a, manifest_a = stratified_trace_split(pool)
        train_b, validation_b, manifest_b = stratified_trace_split(pool)

        self.assertEqual(manifest_a, manifest_b)
        self.assertEqual(train_a.sample_ids, train_b.sample_ids)
        self.assertEqual(validation_a.sample_ids, validation_b.sample_ids)
        self.assertEqual(manifest_a["train_traces"], 160)
        self.assertEqual(manifest_a["validation_traces"], 40)
        self.assertEqual(len(train_a), 480)
        self.assertEqual(len(validation_a), 120)
        self.assertFalse(set(train_a.sample_ids) & set(validation_a.sample_ids))
        self.assertFalse(any("Tokyo" in item["path"] for item in train_a.metadata["source_files"]))
        self.assertFalse(
            any("Tokyo" in item["path"] for item in validation_a.metadata["source_files"])
        )

    def test_sequence_windows_do_not_cross_episode_boundaries(self):
        starts = episode_window_starts(
            [False, False, True, False, False, True],
            max_length=2,
            sample_step=1,
        )
        self.assertEqual(starts, [0, 1, 3, 4])

    def test_sequence_windows_require_final_done(self):
        with self.assertRaisesRegex(ValueError, "final experience"):
            episode_window_starts([False, True, False], max_length=2, sample_step=1)


if __name__ == "__main__":
    unittest.main()
