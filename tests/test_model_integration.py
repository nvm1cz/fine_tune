import unittest
from dataclasses import replace

import numpy as np

from uwsn.models.acoustic import (
    thorp_absorption_db_per_km,
    transmission_loss_db,
)
from uwsn.models.channel import (
    ambient_noise,
    channel_quality,
    rician_ber_from_snr_linear,
    wind_noise_db,
)
from uwsn.models.delay import link_delay_s
from uwsn.models.mobility import gaussian_rbf, update_positions_with_current
from uwsn.routing.transmission import TransmissionStats, _attempt_link
from uwsn.run_config import SIMULATION_PARAMS
from uwsn.models.units import db_to_linear, hz_to_khz, khz_to_hz, linear_to_db


class UnitContractTests(unittest.TestCase):
    def test_frequency_round_trip(self) -> None:
        self.assertEqual(hz_to_khz(10_000.0), 10.0)
        self.assertEqual(khz_to_hz(10.0), 10_000.0)

    def test_db_linear_round_trip(self) -> None:
        self.assertAlmostEqual(linear_to_db(db_to_linear(37.25)), 37.25)


class MobilityContractTests(unittest.TestCase):
    def test_three_dimensional_rbf_and_decay(self) -> None:
        values = gaussian_rbf(
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.asarray([[0.0, 0.0, 0.0]]),
            1.0,
        )
        self.assertAlmostEqual(values[0, 0], 1.0)
        self.assertAlmostEqual(values[1, 0], np.exp(-0.5))

    def test_zero_dt_dead_node_and_boundary_policies(self) -> None:
        positions = np.asarray([[9.0, 5.0, 5.0], [2.0, 2.0, 2.0]])
        base = replace(
            SIMULATION_PARAMS,
            width_m=10.0,
            height_m=10.0,
            depth_m=10.0,
            current_mean_velocity_mps=(2.0, 0.0, 0.0),
            current_horizontal_only=False,
            current_move_dead_nodes=False,
        )
        np.testing.assert_allclose(
            update_positions_with_current(positions, 0.0, 0.0, base),
            positions,
        )
        clipped = update_positions_with_current(
            positions, 0.0, 1.0, base, alive_mask=np.asarray([True, False])
        )
        np.testing.assert_allclose(clipped, [[10.0, 5.0, 5.0], [2.0, 2.0, 2.0]])
        reflected = update_positions_with_current(
            positions[:1], 0.0, 1.0, replace(base, current_boundary_policy="reflect")
        )
        wrapped = update_positions_with_current(
            positions[:1], 0.0, 1.0, replace(base, current_boundary_policy="wrap")
        )
        rejected = update_positions_with_current(
            positions[:1], 0.0, 1.0,
            replace(base, current_boundary_policy="reject_update"),
        )
        self.assertAlmostEqual(reflected[0, 0], 9.0)
        self.assertAlmostEqual(wrapped[0, 0], 1.0)
        self.assertAlmostEqual(rejected[0, 0], 9.0)


class ChannelDelayContractTests(unittest.TestCase):
    def test_paper_transmission_loss_manual_fixture(self) -> None:
        distance_m = 100.0
        frequency_khz = 10.0
        anomaly_db = 2.5
        expected = (
            2.0 * 10.0 * np.log10(distance_m)
            + thorp_absorption_db_per_km(frequency_khz) * 0.1
            + anomaly_db
        )
        actual = transmission_loss_db(
            distance_m,
            frequency_khz,
            2.0,
            model_version="paper_geethu_2017_v1",
            transmission_anomaly_db=anomaly_db,
        )
        self.assertAlmostEqual(actual, expected)

    def test_noise_modes_are_explicit_and_not_double_counted(self) -> None:
        standard = wind_noise_db(10.0, 2.0, "standard_wenz")
        paper = wind_noise_db(10.0, 2.0, "paper_exact")
        self.assertNotEqual(standard, paper)
        approximate = ambient_noise(10.0, 0.5, 2.0, "paper_approximation")
        self.assertAlmostEqual(approximate.total_db, 32.0)

    def test_rician_ber_is_finite_and_monotone_in_snr(self) -> None:
        low = rician_ber_from_snr_linear(1.0, 2.0)
        high = rician_ber_from_snr_linear(10.0, 2.0)
        self.assertTrue(0.0 <= high <= low <= 0.5)

    def test_shannon_limited_bitrate_changes_delay(self) -> None:
        params = replace(
            SIMULATION_PARAMS,
            bitrate_model="shannon_limited",
            modem_max_bitrate_bps=10_000.0,
            capacity_efficiency=0.5,
        )
        slow = link_delay_s(1000, 100.0, params, channel_capacity_bps=1000.0)
        fast = link_delay_s(1000, 100.0, params, channel_capacity_bps=10_000.0)
        self.assertGreater(slow.total_s, fast.total_s)

    def test_mobility_changes_distance_loss_and_propagation_delay(self) -> None:
        params = replace(
            SIMULATION_PARAMS,
            current_mean_velocity_mps=(1.0, 0.0, 0.0),
            source_level_db=180.0,
        )
        start = np.asarray([[10.0, 0.0, 0.0]])
        moved = update_positions_with_current(start, 0.0, 5.0, params)
        d0 = float(np.linalg.norm(start[0]))
        d1 = float(np.linalg.norm(moved[0]))
        self.assertGreater(d1, d0)
        self.assertGreater(
            channel_quality(d1, 10.0, 100, params)["transmission_loss_db"],
            channel_quality(d0, 10.0, 100, params)["transmission_loss_db"],
        )
        self.assertGreater(
            link_delay_s(100, d1, params).propagation_s,
            link_delay_s(100, d0, params).propagation_s,
        )

    def test_sender_dying_stops_retry_loop(self) -> None:
        params = replace(
            SIMULATION_PARAMS,
            source_level_db=-1000.0,
            enable_packet_errors=True,
            enable_retransmission=True,
            max_retries=5,
            minimum_snr_db=None,
            dead_energy_threshold_j=0.0,
        )
        energies = np.asarray([0.15, 1.0])
        stats = TransmissionStats()
        success = _attempt_link(
            params=params,
            packet_bits=100,
            distance_m=10.0,
            tx_energy_j=0.1,
            energies=energies,
            sender=0,
            receiver=1,
            receiver_energy_j=0.01,
            dead_energy_threshold_j=0.0,
            delay_records=[],
            rng=np.random.default_rng(1),
            stats=stats,
        )
        self.assertFalse(success)
        self.assertEqual(stats.attempts, 2)
        self.assertEqual(energies[0], 0.0)


if __name__ == "__main__":
    unittest.main()
