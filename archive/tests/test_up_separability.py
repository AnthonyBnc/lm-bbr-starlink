import unittest

import numpy as np

from analyze_up_separability import (
    centroid_separation_ratio,
    leave_one_trace_out_accuracy,
)


class UpSeparabilityTests(unittest.TestCase):
    def test_trace_held_out_centroids_detect_separable_classes(self):
        features = []
        labels = []
        traces = []
        for trace in range(3):
            for class_index, label in enumerate(range(6, 11)):
                feature = np.zeros(6, dtype=np.float64)
                feature[class_index] = 1.0
                feature[-1] = 0.01 * trace
                features.append(feature)
                labels.append(label)
                traces.append(trace)
        result = leave_one_trace_out_accuracy(
            np.asarray(features), np.asarray(labels), np.asarray(traces)
        )
        self.assertEqual(result["evaluated_samples"], 15)
        self.assertEqual(result["status"], "complete")
        self.assertGreaterEqual(result["accuracy"], 0.8)

    def test_trace_held_out_centroids_report_insufficient_trace_support(self):
        result = leave_one_trace_out_accuracy(
            np.eye(5), np.arange(6, 11), np.zeros(5, dtype=np.int64)
        )
        self.assertEqual(result["status"], "insufficient_trace_support")
        self.assertIsNone(result["accuracy"])

    def test_centroid_ratio_is_positive(self):
        features = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
        labels = np.asarray([6, 6, 7, 7])
        result = centroid_separation_ratio(features, labels)
        self.assertGreater(result["between_to_within_ratio"], 1.0)

if __name__ == "__main__":
    unittest.main()
