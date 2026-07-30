import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from uwsn.cases import SimulationCase
from uwsn.models.channel import (
    awgn_bpsk_ber_from_snr_linear,
    ambient_noise,
    ber_from_snr_linear,
    channel_quality,
    db_to_linear,
    linear_to_db,
    packet_success_probability,
    shipping_noise_db,
    thermal_noise_db,
    transmission_loss_db,
    turbulence_noise_db,
    wind_noise_db,
)
from uwsn.routing.transmission import (
    TransmissionStats,
    _attempt_link,
    aggregate_energy,
    execute_round_transmissions,
    receive_energy,
)
from uwsn.run_config import SIMULATION_PARAMS


class ChannelFormulaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = replace(
            SIMULATION_PARAMS,
            source_level_db=180.0,
            channel_bandwidth_hz=1_000.0,
            shipping_noise_factor=0.5,
            wind_speed_mps=0.0,
        )

    def test_noise_components_follow_underwater_frequency_formulas(self) -> None:
        freq = 10.0
        self.assertAlmostEqual(turbulence_noise_db(freq), -13.0)
        self.assertAlmostEqual(shipping_noise_db(freq, 0.5), 5.922, places=3)
        self.assertAlmostEqual(wind_noise_db(freq, 0.0), 29.319, places=3)
        self.assertAlmostEqual(thermal_noise_db(freq), 5.0)

    def test_noise_psd_components_are_added_in_linear_scale(self) -> None:
        noise = ambient_noise(10.0, shipping_factor=0.5, wind_speed_mps=0.0)
        expected = sum(
            db_to_linear(value)
            for value in (
                noise.turbulence_db,
                noise.shipping_db,
                noise.wind_db,
                noise.thermal_db,
            )
        )
        self.assertAlmostEqual(noise.total_db, linear_to_db(expected))

    def test_wind_and_shipping_inputs_affect_their_own_components(self) -> None:
        self.assertNotEqual(
            wind_noise_db(10.0, 5.0),
            wind_noise_db(10.0, 0.0),
        )
        self.assertNotEqual(
            shipping_noise_db(10.0, 0.5),
            shipping_noise_db(10.0, 0.0),
        )

    def test_transmission_loss_increases_with_distance(self) -> None:
        self.assertGreater(
            transmission_loss_db(200.0, 10.0, 2.0),
            transmission_loss_db(100.0, 10.0, 2.0),
        )

    def test_transmission_loss_increases_with_absorption(self) -> None:
        self.assertGreater(
            transmission_loss_db(1000.0, 100.0, 2.0),
            transmission_loss_db(1000.0, 10.0, 2.0),
        )

    def test_snr_decreases_with_distance(self) -> None:
        near = channel_quality(10.0, 10.0, 100, self.params)
        far = channel_quality(100.0, 10.0, 100, self.params)
        self.assertLess(far["snr_db"], near["snr_db"])

    def test_source_level_enters_sonar_equation_in_db(self) -> None:
        lower = channel_quality(
            100.0, 10.0, 100, replace(self.params, source_level_db=180.0)
        )
        higher = channel_quality(
            100.0, 10.0, 100, replace(self.params, source_level_db=185.0)
        )
        self.assertAlmostEqual(higher["snr_db"] - lower["snr_db"], 5.0)

    def test_snr_decreases_when_ambient_noise_increases(self) -> None:
        quiet = channel_quality(
            100.0,
            10.0,
            100,
            replace(self.params, shipping_noise_factor=0.0, wind_speed_mps=0.0),
        )
        noisy = channel_quality(
            100.0,
            10.0,
            100,
            replace(self.params, shipping_noise_factor=1.0, wind_speed_mps=10.0),
        )
        self.assertLess(noisy["snr_db"], quiet["snr_db"])

    def test_ber_increases_when_snr_decreases(self) -> None:
        self.assertGreater(ber_from_snr_linear(1.0), ber_from_snr_linear(100.0))

    def test_none_fading_uses_awgn_bpsk_without_fading_functions(self) -> None:
        params = replace(self.params, fading_model="none")
        with patch("uwsn.models.channel.ber_from_snr_linear") as rayleigh, patch(
            "uwsn.models.channel.rician_ber_from_snr_linear"
        ) as rician:
            actual = channel_quality(100.0, 10.0, 100, params)
        rayleigh.assert_not_called()
        rician.assert_not_called()
        self.assertAlmostEqual(
            actual["ber"],
            awgn_bpsk_ber_from_snr_linear(actual["snr_linear"]),
        )

    def test_success_threshold_does_not_overwrite_physical_probability(self) -> None:
        unthresholded = channel_quality(
            100.0,
            10.0,
            6400,
            replace(self.params, minimum_success_probability=None),
        )
        thresholded = channel_quality(
            100.0,
            10.0,
            6400,
            replace(self.params, minimum_success_probability=0.95),
        )
        self.assertEqual(
            thresholded["packet_success_probability"],
            unthresholded["packet_success_probability"],
        )
        self.assertEqual(
            thresholded["success_threshold_passed"],
            thresholded["packet_success_probability"] >= 0.95,
        )

    def test_success_decreases_with_ber_and_packet_length(self) -> None:
        self.assertLess(
            packet_success_probability(0.1, 100),
            packet_success_probability(0.01, 100),
        )
        self.assertLess(
            packet_success_probability(0.01, 1000),
            packet_success_probability(0.01, 100),
        )

    def test_bandwidth_converts_psd_to_power_exactly_once(self) -> None:
        one_hz = channel_quality(
            100.0, 10.0, 100, replace(self.params, channel_bandwidth_hz=1.0)
        )
        thousand_hz = channel_quality(
            100.0, 10.0, 100, replace(self.params, channel_bandwidth_hz=1000.0)
        )
        self.assertAlmostEqual(
            thousand_hz["noise_power_db"] - one_hz["noise_power_db"],
            30.0,
        )
        self.assertAlmostEqual(
            thousand_hz["noise_psd_db"],
            one_hz["noise_psd_db"],
        )

    def test_channel_quality_contains_required_outputs(self) -> None:
        actual = channel_quality(100.0, 10.0, 6400, self.params)
        required = {
            "distance_m",
            "frequency_khz",
            "absorption_db_per_km",
            "attenuation_linear",
            "transmission_loss_db",
            "noise_psd_db",
            "noise_power_db",
            "snr_db",
            "snr_linear",
            "ber",
            "per",
            "packet_success_probability",
            "channel_capacity_bps",
        }
        self.assertTrue(required.issubset(actual))
        self.assertTrue(0.0 <= actual["ber"] <= 0.5)
        self.assertTrue(0.0 <= actual["packet_success_probability"] <= 1.0)


class ChannelIntegrationTests(unittest.TestCase):
    def _failed_link(self, retransmission: bool) -> tuple[np.ndarray, list[float], TransmissionStats]:
        params = replace(
            SIMULATION_PARAMS,
            source_level_db=-1000.0,
            channel_bandwidth_hz=1000.0,
            enable_packet_errors=True,
            enable_retransmission=retransmission,
            max_retries=3,
            minimum_snr_db=None,
            dead_energy_threshold_j=0.0,
        )
        energies = np.asarray([10.0, 10.0])
        delays: list[float] = []
        stats = TransmissionStats()
        _attempt_link(
            params=params,
            packet_bits=100,
            distance_m=100.0,
            tx_energy_j=0.1,
            energies=energies,
            sender=0,
            receiver=1,
            receiver_energy_j=0.01,
            dead_energy_threshold_j=0.0,
            delay_records=delays,
            rng=np.random.default_rng(7),
            stats=stats,
        )
        return energies, delays, stats

    def test_same_seed_produces_same_attempt_result(self) -> None:
        first = self._failed_link(True)
        second = self._failed_link(True)
        np.testing.assert_allclose(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertEqual(first[2], second[2])

    def test_retry_increases_physical_energy_and_delay(self) -> None:
        no_retry_energy, no_retry_delay, no_retry_stats = self._failed_link(False)
        retry_energy, retry_delay, retry_stats = self._failed_link(True)
        self.assertLess(retry_energy.sum(), no_retry_energy.sum())
        self.assertGreater(sum(retry_delay), sum(no_retry_delay))
        self.assertEqual(no_retry_stats.attempts, 1)
        self.assertEqual(retry_stats.attempts, 4)
        self.assertEqual(retry_stats.retransmissions, 3)

    def test_two_failed_attempts_charge_tx_rx_and_delay_twice(self) -> None:
        params = replace(
            SIMULATION_PARAMS,
            source_level_db=-1000.0,
            channel_bandwidth_hz=1000.0,
            enable_packet_errors=True,
            enable_retransmission=True,
            max_retries=1,
            minimum_snr_db=None,
            dead_energy_threshold_j=0.0,
        )
        energies = np.asarray([10.0, 10.0])
        delays: list[float] = []
        stats = TransmissionStats()
        result = _attempt_link(
            params=params,
            packet_bits=100,
            distance_m=100.0,
            tx_energy_j=0.1,
            energies=energies,
            sender=0,
            receiver=1,
            receiver_energy_j=0.01,
            dead_energy_threshold_j=0.0,
            delay_records=delays,
            rng=np.random.default_rng(7),
            stats=stats,
        )
        self.assertFalse(result)
        self.assertEqual(stats.attempts, 2)
        self.assertEqual(stats.retransmissions, 1)
        self.assertAlmostEqual(energies[0], 10.0 - 2 * 0.1)
        self.assertAlmostEqual(energies[1], 10.0 - 2 * 0.01)
        self.assertEqual(len(delays), 2)
        self.assertAlmostEqual(delays[0], delays[1])

    def test_modem_transition_delay_is_charged_per_attempt(self) -> None:
        _, baseline_delays, _ = self._failed_link(True)
        params = replace(
            SIMULATION_PARAMS,
            source_level_db=-1000.0,
            channel_bandwidth_hz=1000.0,
            enable_packet_errors=True,
            enable_retransmission=True,
            max_retries=3,
            modem_transition_delay_s=1.5,
            dead_energy_threshold_j=0.0,
        )
        energies = np.asarray([10.0, 10.0])
        delays: list[float] = []
        stats = TransmissionStats()
        _attempt_link(
            params=params,
            packet_bits=100,
            distance_m=100.0,
            tx_energy_j=0.1,
            energies=energies,
            sender=0,
            receiver=1,
            receiver_energy_j=0.01,
            dead_energy_threshold_j=0.0,
            delay_records=delays,
            rng=np.random.default_rng(7),
            stats=stats,
        )
        self.assertEqual(len(delays), 4)
        self.assertAlmostEqual(
            sum(delays) - sum(baseline_delays),
            4 * 1.5,
        )

    def test_aggregation_is_not_charged_for_failed_member_packet(self) -> None:
        case = SimulationCase("channel", initial_energy=10.0, packet_size_bits=100, node_count=2)
        params = replace(
            SIMULATION_PARAMS,
            source_level_db=-1000.0,
            channel_bandwidth_hz=1000.0,
            enable_packet_errors=True,
            enable_retransmission=False,
            minimum_snr_db=None,
            dead_energy_threshold_j=0.0,
            transmission_range_m=1000.0,
            broadcast_packet_size_bits=0,
        )
        distance_matrix = np.asarray([[0.0, 10.0], [10.0, 0.0]])
        tx_cost_matrix = np.asarray([[0.0, 0.001], [0.001, 0.0]])
        tx_sink = np.asarray([0.001, 0.001])

        def run(assignments: dict[int, list[int]]) -> np.ndarray:
            energies = np.asarray([10.0, 10.0])
            execute_round_transmissions(
                case,
                params,
                distance_matrix,
                tx_cost_matrix,
                tx_sink,
                energies,
                np.asarray([1, 1]),
                np.asarray([10.0, 10.0]),
                assignments,
                [0],
                0,
                0.0,
                rng=np.random.default_rng(1),
            )
            return energies

        without_member = run({0: [0]})
        with_failed_member = run({0: [0, 1]})
        extra_ch_cost = without_member[0] - with_failed_member[0]
        self.assertAlmostEqual(extra_ch_cost, receive_energy(params, 100))
        self.assertNotAlmostEqual(extra_ch_cost, receive_energy(params, 100) + aggregate_energy(params, 100))

    def test_disabled_packet_errors_matches_legacy_behavior(self) -> None:
        case = SimulationCase("legacy", initial_energy=10.0, packet_size_bits=100, node_count=1)
        params = replace(
            SIMULATION_PARAMS,
            enable_packet_errors=False,
            dead_energy_threshold_j=0.0,
            transmission_range_m=1000.0,
            broadcast_packet_size_bits=0,
        )
        args = (
            case,
            params,
            np.asarray([[0.0]]),
            np.asarray([[0.0]]),
            np.asarray([0.001]),
        )
        first = np.asarray([10.0])
        second = np.asarray([10.0])
        received_first = execute_round_transmissions(
            *args,
            first,
            np.asarray([1]),
            np.asarray([10.0]),
            {0: [0]},
            [0],
            0,
            0.0,
            rng=np.random.default_rng(1),
        )
        received_second = execute_round_transmissions(
            *args,
            second,
            np.asarray([1]),
            np.asarray([10.0]),
            {0: [0]},
            [0],
            0,
            0.0,
            rng=np.random.default_rng(999),
        )
        self.assertEqual(received_first, received_second)
        np.testing.assert_allclose(first, second)


if __name__ == "__main__":
    unittest.main()
