import inspect
import unittest
from dataclasses import replace

import numpy as np

import uwsn.models.channel as channel_module
import uwsn.routing.transmission as routing_module
from uwsn.models.acoustic import (
    attenuation_linear,
    attenuation_linear_array,
    thorp_absorption_db_per_km,
    transmission_loss_db,
)
from uwsn.models.channel import channel_quality
from uwsn.models.energy import attenuation, transmit_energy_j
from uwsn.run_config import SIMULATION_PARAMS


class AcousticPropagationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = replace(
            SIMULATION_PARAMS,
            acoustic_frequency_khz=10.0,
            spreading_factor=2.0,
            source_level_db=180.0,
            channel_bandwidth_hz=1000.0,
        )

    def test_thorp_is_finite_positive_and_frequency_sensitive(self) -> None:
        values = [thorp_absorption_db_per_km(f) for f in (1.0, 10.0, 100.0)]
        self.assertTrue(all(np.isfinite(value) and value > 0.0 for value in values))
        self.assertGreater(values[-1], values[1])

    def test_attenuation_and_loss_are_equivalent(self) -> None:
        value = attenuation_linear(100.0, 10.0, 2.0)
        self.assertGreater(value, 0.0)
        self.assertAlmostEqual(
            transmission_loss_db(100.0, 10.0, 2.0),
            10.0 * np.log10(value),
            places=12,
        )

    def test_scalar_and_array_use_the_same_logic(self) -> None:
        distances = np.asarray([0.0, 10.0, 50.0, 100.0, 500.0, 1000.0])
        array = attenuation_linear_array(distances, 10.0, 2.0)
        scalar = np.asarray(
            [attenuation_linear(value, 10.0, 2.0) for value in distances]
        )
        np.testing.assert_allclose(array, scalar, rtol=0.0, atol=0.0)

    def test_energy_and_channel_use_shared_attenuation(self) -> None:
        shared = attenuation_linear(
            100.0,
            10.0,
            2.0,
            model_version=self.params.acoustic_model_version,
            transmission_anomaly_db=self.params.transmission_anomaly_db,
        )
        self.assertAlmostEqual(attenuation(100.0, self.params), shared)
        quality = channel_quality(100.0, 10.0, 6400, self.params)
        self.assertAlmostEqual(quality["attenuation_linear"], shared)
        self.assertAlmostEqual(
            quality["transmission_loss_db"], 10.0 * np.log10(shared)
        )

    def test_no_private_spreading_or_tx_energy_formula_remains(self) -> None:
        channel_source = inspect.getsource(channel_module)
        routing_source = inspect.getsource(routing_module)
        self.assertNotIn("distance**spreading", channel_source)
        self.assertNotIn("tx_scale =", routing_source)
        self.assertIn("transmit_energy_j(", routing_source)

    def test_tx_energy_contains_attenuation_once(self) -> None:
        bits = 6400
        base = (
            bits
            * self.params.energy_calibration_factor
            * self.params.transmit_power_p0
            * self.params.electronic_energy_e1_nj
            * 1.0e-9
        )
        actual = transmit_energy_j(100.0, bits, self.params)
        self.assertAlmostEqual(
            actual / base,
            attenuation_linear(
                100.0,
                10.0,
                2.0,
                model_version=self.params.acoustic_model_version,
                transmission_anomaly_db=self.params.transmission_anomaly_db,
            ),
        )

    def test_benchmark_distance_loss_is_non_decreasing(self) -> None:
        distances = (10.0, 50.0, 100.0, 500.0, 1000.0)
        losses = [transmission_loss_db(d, 10.0, 2.0) for d in distances]
        self.assertTrue(all(left <= right for left, right in zip(losses, losses[1:])))

    def test_invalid_values_are_rejected(self) -> None:
        for args in (
            (-1.0, 10.0, 2.0),
            (1.0, 0.0, 2.0),
            (1.0, 10.0, 0.0),
        ):
            with self.assertRaises(ValueError):
                attenuation_linear(*args)

    def test_manual_100m_fixture(self) -> None:
        distance_m = 100.0
        frequency_khz = self.params.acoustic_frequency_khz
        distance_km = distance_m / 1000.0
        alpha = thorp_absorption_db_per_km(frequency_khz)
        spreading_term = distance_km**self.params.spreading_factor
        absorption_term = 10.0 ** (alpha * distance_km / 10.0)
        attenuation_value = attenuation_linear(
            distance_m, frequency_khz, self.params.spreading_factor
        )
        quality = channel_quality(distance_m, frequency_khz, 6400, self.params)
        fixture = {
            "distance_km": distance_km,
            "alpha_db_per_km": alpha,
            "spreading_term": spreading_term,
            "absorption_term": absorption_term,
            "attenuation_linear": attenuation_value,
            "transmission_loss_db": quality["transmission_loss_db"],
            "tx_energy_j": transmit_energy_j(distance_m, 6400, self.params),
            "snr_db": quality["snr_db"],
            "ber": quality["ber"],
            "per": quality["per"],
        }
        print(f"manual_acoustic_fixture={fixture}")
        self.assertTrue(all(np.isfinite(value) for value in fixture.values()))


if __name__ == "__main__":
    unittest.main()
