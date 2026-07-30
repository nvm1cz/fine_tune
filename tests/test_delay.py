import unittest
from dataclasses import replace

from uwsn.models.delay import (
    eegnbr_forwarding_delay_without_queue_s,
    link_delay_s,
    propagation_delay_s,
    transmission_delay_s,
)
from uwsn.run_config import SIMULATION_PARAMS


class DelayModelTests(unittest.TestCase):
    def test_transmission_delay_is_packet_size_over_bit_rate(self) -> None:
        self.assertAlmostEqual(transmission_delay_s(6400, 3200.0), 2.0)

    def test_propagation_delay_is_distance_over_sound_speed(self) -> None:
        self.assertAlmostEqual(propagation_delay_s(1500.0, 1500.0), 1.0)

    def test_link_delay_sums_tx_rx_propagation_queue_and_holding(self) -> None:
        params = replace(
            SIMULATION_PARAMS,
            node_bit_rate_bps=3200.0,
            sound_speed_water_mps=1500.0,
            byte_alignment_delay_s=0.25,
            holding_time_s=0.75,
        )

        actual = link_delay_s(6400, 1500.0, params)

        self.assertAlmostEqual(actual.transmission_s, 2.0)
        self.assertAlmostEqual(actual.reception_s, 2.0)
        self.assertAlmostEqual(actual.propagation_s, 1.0)
        self.assertAlmostEqual(actual.total_s, 6.0)

    def test_eegnbr_delay_without_queue_matches_timer_plus_two_tr_plus_pr(self) -> None:
        params = replace(
            SIMULATION_PARAMS,
            node_bit_rate_bps=3200.0,
            sound_speed_water_mps=1500.0,
            query_timer_s=0.5,
        )

        actual = eegnbr_forwarding_delay_without_queue_s(6400, 1500.0, params)

        self.assertAlmostEqual(actual, 5.5)


if __name__ == "__main__":
    unittest.main()
