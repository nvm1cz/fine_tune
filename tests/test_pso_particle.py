import unittest

import numpy as np

from uwsn.optimizers.common import _decode_priority_vector, _target_cluster_head_count


class PsoParticleDecodingTests(unittest.TestCase):
    def test_target_count_uses_half_up_rounding_and_candidate_cap(self) -> None:
        self.assertEqual(_target_cluster_head_count(20, 8, 0.20), 4)
        self.assertEqual(_target_cluster_head_count(5, 5, 0.50), 3)
        self.assertEqual(_target_cluster_head_count(100, 3, 0.20), 3)
        self.assertEqual(_target_cluster_head_count(1, 1, 0.01), 1)
        self.assertEqual(_target_cluster_head_count(0, 0, 0.20), 0)

    def test_decoder_selects_exactly_the_top_k_candidates(self) -> None:
        priorities = np.asarray([0.20, 0.90, 0.40, 0.80])
        candidates = np.asarray([10, 11, 12, 13])

        selected = _decode_priority_vector(priorities, candidates, 2)

        self.assertEqual(selected, [11, 13])

    def test_decoder_rejects_a_dimension_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            _decode_priority_vector(np.asarray([0.1, 0.2]), np.asarray([1]), 1)


if __name__ == "__main__":
    unittest.main()
