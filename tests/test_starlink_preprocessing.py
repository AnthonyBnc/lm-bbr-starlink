import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

from utils.bbr import BW_CRUISE, BW_DOWN, BW_UP, action_index_to_gain, validate_phase_action
from utils.starlink_preprocessing import (
    PaperPreprocessingConfig,
    build_experience_pool,
    build_held_out_experience_pool,
    build_paper_labels,
    detect_bbr_phases,
    infer_trace_identity,
    infer_held_out_trace_identity,
    parse_iperf3_intervals,
)


def _row(index, throughput, retransmits=0, rtt=100):
    return {
        "end": float(index + 1),
        "bits_per_second": float(throughput),
        "retransmits": float(retransmits),
        "snd_cwnd": 1000.0,
        "snd_wnd": 2000.0,
        "rtt": float(rtt),
        "rttvar": 5.0,
    }


class StarlinkPreprocessingTests(unittest.TestCase):
    def test_paper_window_is_twenty_one_samples(self):
        config = PaperPreprocessingConfig()
        self.assertEqual(config.half_window, 10)

    def test_phase_detector_marks_phase_safe_events(self):
        throughput = [10.0] * 25
        throughput[6] = 30.0
        throughput[10] = 1.0
        phases = detect_bbr_phases(throughput)
        self.assertEqual(phases[6], BW_UP)
        self.assertEqual(phases[10], BW_DOWN)
        self.assertEqual(phases[11:17], [BW_CRUISE] * 6)

    def test_generated_labels_are_valid_for_every_phase(self):
        throughputs = [10.0] * 25
        throughputs[6] = 30.0
        throughputs[10] = 1.0
        rows = [_row(index, value, retransmits=index % 3, rtt=100 + index) for index, value in enumerate(throughputs)]
        phases, labels, rewards = build_paper_labels(rows)
        self.assertEqual(len(phases), len(labels))
        self.assertTrue(all(np.isfinite(rewards)))
        for phase, label in zip(phases, labels):
            validate_phase_action(phase, label)

    def test_equation_2_3_labels_use_granular_up_actions(self):
        rows = [_row(index, 10.0) for index in range(25)]
        rows[6]["bits_per_second"] = 30.0
        rows[10]["bits_per_second"] = 1.0
        rows[20]["bits_per_second"] = 100.0
        phases, labels, _ = build_paper_labels(rows)
        up_gain = action_index_to_gain(labels[6])
        down_gain = action_index_to_gain(labels[10])
        self.assertEqual(phases[6], BW_UP)
        self.assertEqual(up_gain, 1.25)
        self.assertEqual(phases[10], BW_DOWN)
        self.assertEqual(down_gain, 0.90)

    def test_equation_2_3_labels_can_pick_midrange_up_gain(self):
        rows = [_row(index, 10.0) for index in range(25)]
        rows[6]["bits_per_second"] = 80.0
        rows[10]["bits_per_second"] = 1.0
        rows[20]["bits_per_second"] = 100.0
        phases, labels, _ = build_paper_labels(rows)
        self.assertEqual(phases[6], BW_UP)
        self.assertEqual(action_index_to_gain(labels[6]), 1.05)

    def test_parser_tolerates_trailing_diagnostics(self):
        stream = dict(_row(0, 42.0), sender=True)
        document = {"intervals": [{"streams": [stream]}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(document) + "\niperf Done.\n", encoding="utf-8")
            rows, leading, trailing = parse_iperf3_intervals(path)
        self.assertEqual(rows[0]["bits_per_second"], 42.0)
        self.assertEqual(leading, "")
        self.assertIn("iperf Done", trailing)

    def test_parser_tolerates_leading_diagnostics(self):
        stream = dict(_row(0, 42.0), sender=True)
        document = {"intervals": [{"streams": [stream]}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(
                "iperf_json_finish: pthread_mutex_lock: Success\n"
                + json.dumps(document),
                encoding="utf-8",
            )
            rows, leading, trailing = parse_iperf3_intervals(path)
        self.assertEqual(rows[0]["bits_per_second"], 42.0)
        self.assertIn("iperf_json_finish", leading)
        self.assertEqual(trailing, "")

    def test_tokyo_is_rejected_before_file_read(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            tokyo_path = raw_root / "downlink-sequential-logs" / "Tokyo" / "missing.json"
            with self.assertRaisesRegex(ValueError, "Tokyo is held out"):
                infer_trace_identity(tokyo_path, raw_root)

    def test_held_out_identity_accepts_only_tokyo(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            tokyo_path = raw_root / "downlink-sequential-logs" / "Tokyo" / "missing.json"
            relative, location, stream = infer_held_out_trace_identity(tokyo_path, raw_root)
            self.assertEqual(location, "Tokyo")
            self.assertEqual(stream, "downlink-sequential-logs")
            self.assertIn("Tokyo", relative.parts)
            ohio_path = raw_root / "downlink-sequential-logs" / "Ohio" / "missing.json"
            with self.assertRaisesRegex(ValueError, "Tokyo only"):
                infer_held_out_trace_identity(ohio_path, raw_root)

    def test_held_out_pool_requires_non_aliasing_explicit_location_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            path = raw_root / "downlink-sequential-logs" / "Tokyo" / "bbr_Tokyo__REV_run1.json"
            path.parent.mkdir(parents=True)
            intervals = [
                {"streams": [dict(_row(index, 10.0), sender=True)]}
                for index in range(25)
            ]
            path.write_text(json.dumps({"intervals": intervals}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not alias"):
                build_held_out_experience_pool([path], raw_root, 4)
            pool = build_held_out_experience_pool([path], raw_root, 5)
        self.assertEqual(pool.metadata["split_role"], "held_out_test")
        self.assertEqual(pool.metadata["location_flags"], {"Tokyo": 5})
        self.assertTrue(all(state[0] == 5 for state in pool.states))

    def test_pool_has_unique_ids_and_nine_state_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            path = raw_root / "downlink-sequential-logs" / "Ohio" / "bbr_Ohio__REV_run1.json"
            path.parent.mkdir(parents=True)
            intervals = []
            for index in range(25):
                stream = dict(_row(index, 10.0 + (20.0 if index == 6 else 0.0)), sender=True)
                intervals.append({"streams": [stream]})
            path.write_text(json.dumps({"intervals": intervals}), encoding="utf-8")
            pool = build_experience_pool([path], raw_root)
        self.assertEqual(len(pool), 25)
        self.assertEqual(len(pool.states[0]), 9)
        self.assertEqual(len(set(pool.sample_ids)), len(pool))
        self.assertTrue(pool.dones[-1])
        self.assertFalse(any(pool.dones[:-1]))

    def test_pool_pickle_round_trip_preserves_done_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            path = raw_root / "downlink-sequential-logs" / "Ohio" / "bbr_Ohio__REV_run1.json"
            path.parent.mkdir(parents=True)
            intervals = []
            for index in range(3):
                stream = dict(_row(index, 10.0), sender=True)
                intervals.append({"streams": [stream]})
            path.write_text(json.dumps({"intervals": intervals}), encoding="utf-8")
            pool = build_experience_pool([path], raw_root)
            restored = pickle.loads(pickle.dumps(pool))
        self.assertEqual(restored.dones, [False, False, True])


if __name__ == "__main__":
    unittest.main()
