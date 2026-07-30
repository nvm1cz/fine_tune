import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from uwsn.models.mobility import (
    gaussian_rbf,
    rbf_velocity_field,
    rope_tension_velocity,
    tidal_current_velocity,
    update_positions_with_current,
)
from uwsn.run_config import SIMULATION_PARAMS
from uwsn.cases import SimulationCase
from uwsn.simulator import UWSNSimulator


class MobilityTests(unittest.TestCase):
    def test_tidal_current_velocity_uses_mean_cos_and_sin_terms(self) -> None:
        params = replace(
            SIMULATION_PARAMS,
            current_mean_velocity_mps=(0.10, 0.20, 0.30),
            current_cos_amplitude_mps=(0.01, 0.02, 0.03),
            current_sin_amplitude_mps=(0.04, 0.05, 0.06),
            current_angular_frequency_rad_s=np.pi / 2,
        )

        actual = tidal_current_velocity(1.0, params)

        np.testing.assert_allclose(actual, np.asarray([0.14, 0.25, 0.36]))

    def test_update_positions_can_keep_depth_fixed_and_clip_bounds(self) -> None:
        params = replace(
            SIMULATION_PARAMS,
            width_m=10.0,
            height_m=10.0,
            depth_m=10.0,
            current_mean_velocity_mps=(2.0, -3.0, 5.0),
            current_horizontal_only=True,
        )
        positions = np.asarray([[9.0, 1.0, 4.0]])

        actual = update_positions_with_current(positions, t_s=0.0, dt_s=2.0, params=params)

        np.testing.assert_allclose(actual, np.asarray([[10.0, 0.0, 4.0]]))

    def test_rbf_velocity_field_expands_position_dependent_current_terms(self) -> None:
        tau = np.asarray([[0.0, 0.0], [1.0, 0.0]])
        centers = np.asarray([[0.0, 0.0]])
        coeffs = np.asarray([[2.0, 4.0, 6.0]])

        phi = gaussian_rbf(tau, centers, sigma_m=1.0)
        actual = rbf_velocity_field(tau, centers, coeffs, sigma_m=1.0)

        np.testing.assert_allclose(phi[:, 0], np.asarray([1.0, np.exp(-0.5)]))
        np.testing.assert_allclose(actual[0], np.asarray([2.0, 4.0, 6.0]))
        np.testing.assert_allclose(actual[1], np.asarray([2.0, 4.0, 6.0]) * np.exp(-0.5))

    def test_rbf_tidal_current_velocity_supports_multiple_tidal_components(self) -> None:
        params = replace(
            SIMULATION_PARAMS,
            current_model="rbf",
            current_rbf_centers_xy_m=((0.0, 0.0),),
            current_rbf_sigma_m=1.0,
            current_rbf_mean_coefficients_mps=((0.10, 0.20, 0.30),),
            current_rbf_cos_coefficients_mps=(
                ((0.01, 0.02, 0.03),),
                ((0.10, 0.00, 0.00),),
            ),
            current_rbf_sin_coefficients_mps=(
                ((0.04, 0.05, 0.06),),
                ((0.00, 0.20, 0.00),),
            ),
            current_rbf_angular_frequencies_rad_s=(np.pi / 2, np.pi),
        )
        positions = np.asarray([[0.0, 0.0, 5.0]])

        actual = tidal_current_velocity(1.0, params, positions)

        np.testing.assert_allclose(actual, np.asarray([[0.04, 0.25, 0.36]]))

    def test_rope_tension_velocity_uses_horizontal_force_component(self) -> None:
        params = replace(
            SIMULATION_PARAMS,
            current_include_rope_tension_velocity=True,
            node_mass_kg=2.0,
            node_buoyancy_force_n=8.0,
            node_gravity_force_n=2.0,
            node_rope_tension_n=10.0,
            node_rope_horizontal_direction=(3.0, 4.0, 0.0),
        )

        actual = rope_tension_velocity(dt_s=0.5, params=params)

        # F_horizontal = sqrt(10^2 - 6^2) = 8 N, acceleration = 4 m/s^2.
        np.testing.assert_allclose(actual, np.asarray([1.2, 1.6, 0.0]))

    def test_simulator_updates_tidal_rbf_every_two_seconds_only(self) -> None:
        params = replace(
            SIMULATION_PARAMS,
            current_model_enabled=True,
            current_model="rbf",
            current_round_duration_s=1.0,
            current_update_interval_rounds=2,
            current_move_dead_nodes=False,
        )
        simulator = UWSNSimulator(
            SimulationCase("mobility", 0.5, 100, node_count=2),
            params,
            seed=4,
            verbose=False,
        )
        fixed_sink = simulator.sink.copy()
        simulator.energies[1] = 0.0
        original = simulator.positions.copy()

        def move(positions, t_s, dt_s, _params, alive_mask):
            self.assertEqual(t_s, 2.0)
            self.assertEqual(dt_s, 2.0)
            moved = positions.copy()
            moved[alive_mask] += 1.0
            return moved

        with patch("uwsn.simulator.update_positions_with_current", side_effect=move) as update:
            simulator._apply_current_model(1)
            update.assert_not_called()
            simulator._apply_current_model(2)
            update.assert_called_once()

        np.testing.assert_allclose(simulator.sink, fixed_sink)
        np.testing.assert_allclose(simulator.positions[0], original[0] + 1.0)
        np.testing.assert_allclose(simulator.positions[1], original[1])


if __name__ == "__main__":
    unittest.main()
