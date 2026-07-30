from __future__ import annotations

import numpy as np

from ..run_config import TunableParams


def gaussian_rbf(
    positions_m: np.ndarray,
    centers_m: np.ndarray,
    sigma_m: float,
) -> np.ndarray:
    """Evaluate phi_j(tau) = exp(-||tau - c_j||^2 / (2 sigma^2))."""
    tau = np.asarray(positions_m, dtype=float)
    centers = np.asarray(centers_m, dtype=float)
    if tau.ndim != 2 or tau.shape[1] not in (2, 3):
        raise ValueError("RBF positions must have shape (N, 2) or (N, 3)")
    if centers.size == 0:
        return np.zeros((tau.shape[0], 0), dtype=float)
    if centers.ndim != 2 or centers.shape[1] != tau.shape[1]:
        raise ValueError("RBF centers and positions must have the same spatial dimension")

    sigma = float(sigma_m)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("rbf_sigma_m must be finite and positive")
    deltas = tau[:, None, :] - centers[None, :, :]
    squared_distances = np.sum(deltas * deltas, axis=2)
    return np.exp(-squared_distances / (2.0 * sigma * sigma))


def rbf_velocity_field(
    positions_m: np.ndarray,
    centers_m: np.ndarray,
    coefficients: np.ndarray,
    sigma_m: float,
) -> np.ndarray:
    """Expand a position-dependent velocity term with Gaussian RBFs."""
    tau = np.asarray(positions_m, dtype=float)
    coeffs = np.asarray(coefficients, dtype=float)
    if coeffs.size == 0:
        return np.zeros((tau.shape[0], 3), dtype=float)
    if coeffs.ndim != 2 or coeffs.shape[1] != 3:
        raise ValueError("RBF coefficients must have shape (M, 3)")

    phi = gaussian_rbf(tau, centers_m, sigma_m)
    if phi.shape[1] != coeffs.shape[0]:
        raise ValueError("RBF centers and coefficients must use the same M")
    return phi @ coeffs


def _uniform_tidal_current_velocity(t_s: float, params: TunableParams) -> np.ndarray:
    mean = np.asarray(params.current_mean_velocity_mps, dtype=float)
    amp_cos = np.asarray(params.current_cos_amplitude_mps, dtype=float)
    amp_sin = np.asarray(params.current_sin_amplitude_mps, dtype=float)
    omega = float(params.current_angular_frequency_rad_s)
    return mean + amp_cos * np.cos(omega * t_s) + amp_sin * np.sin(omega * t_s)


def _rbf_tidal_current_velocity(positions: np.ndarray, t_s: float, params: TunableParams) -> np.ndarray:
    spatial_positions = np.asarray(positions, dtype=float)
    configured_3d = tuple(params.current_rbf_centers_m)
    if configured_3d:
        centers = np.asarray(configured_3d, dtype=float)
    else:
        # Backward-compatible 2-D coefficient files remain supported explicitly.
        spatial_positions = spatial_positions[:, :2]
        centers = np.asarray(params.current_rbf_centers_xy_m, dtype=float)
    sigma = float(params.current_rbf_sigma_m)
    mean = rbf_velocity_field(
        spatial_positions,
        centers,
        np.asarray(params.current_rbf_mean_coefficients_mps, dtype=float),
        sigma,
    )

    velocity = mean
    cos_coefficients = tuple(params.current_rbf_cos_coefficients_mps)
    sin_coefficients = tuple(params.current_rbf_sin_coefficients_mps)
    omegas = tuple(params.current_rbf_angular_frequencies_rad_s)
    if not (len(cos_coefficients) == len(sin_coefficients) == len(omegas)):
        raise ValueError("RBF tidal components must have matching cos/sin/omega lengths")

    for cos_coeff, sin_coeff, omega in zip(cos_coefficients, sin_coefficients, omegas):
        velocity = velocity + (
            rbf_velocity_field(spatial_positions, centers, np.asarray(cos_coeff, dtype=float), sigma)
            * np.cos(float(omega) * t_s)
        )
        velocity = velocity + (
            rbf_velocity_field(spatial_positions, centers, np.asarray(sin_coeff, dtype=float), sigma)
            * np.sin(float(omega) * t_s)
        )
    return velocity


def tidal_current_velocity(
    t_s: float,
    params: TunableParams,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    """Return current velocity from the uniform or Jiang-Xu RBF tidal model."""
    model = str(getattr(params, "current_model", "uniform")).lower()
    if positions is None or model == "uniform":
        return _uniform_tidal_current_velocity(t_s, params)
    if model == "rbf":
        return _rbf_tidal_current_velocity(np.asarray(positions, dtype=float), t_s, params)
    raise ValueError("Unsupported current_model. Use 'uniform' or 'rbf'.")


def rope_tension_velocity(dt_s: float, params: TunableParams) -> np.ndarray:
    """Return the optional horizontal velocity term from rope tension acceleration."""
    if not bool(params.current_include_rope_tension_velocity):
        return np.zeros(3, dtype=float)

    mass = float(params.node_mass_kg)
    if mass <= 0.0:
        raise ValueError("node_mass_kg must be positive when rope tension velocity is enabled")

    buoyancy_minus_gravity = float(params.node_buoyancy_force_n) - float(params.node_gravity_force_n)
    tension = float(params.node_rope_tension_n)
    horizontal_force_squared = max(tension * tension - buoyancy_minus_gravity * buoyancy_minus_gravity, 0.0)
    horizontal_force = np.sqrt(horizontal_force_squared)

    direction = np.asarray(params.node_rope_horizontal_direction, dtype=float)
    direction[2] = 0.0
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("node_rope_horizontal_direction must be non-zero")

    horizontal_acceleration = horizontal_force / mass
    return (direction / norm) * horizontal_acceleration * float(dt_s)


def update_positions_with_current(
    positions: np.ndarray,
    t_s: float,
    dt_s: float,
    params: TunableParams,
    alive_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the Euler update p(t + dt) = p(t) + v_node(t) dt."""
    updated = np.asarray(positions, dtype=float).copy()
    velocity = tidal_current_velocity(t_s, params, updated)
    velocity = velocity + rope_tension_velocity(dt_s, params)

    if bool(params.current_horizontal_only):
        velocity = velocity.copy()
        if velocity.ndim == 1:
            velocity[2] = 0.0
        else:
            velocity[:, 2] = 0.0

    dt = float(dt_s)
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("mobility dt_s must be finite and non-negative")
    displacement = np.broadcast_to(velocity, updated.shape).copy() * dt
    if alive_mask is not None and not bool(params.current_move_dead_nodes):
        mask = np.asarray(alive_mask, dtype=bool)
        if mask.shape != (updated.shape[0],):
            raise ValueError("alive_mask must have shape (N,)")
        displacement = displacement.copy()
        displacement[~mask] = 0.0
    proposed = updated + displacement

    lower = np.zeros(3, dtype=float)
    upper = np.asarray([params.width_m, params.height_m, params.depth_m], dtype=float)
    policy = str(params.current_boundary_policy).lower()
    if policy == "clip":
        return np.clip(proposed, lower, upper)
    if policy == "wrap":
        return np.mod(proposed, upper)
    if policy == "reject_update":
        invalid = np.any((proposed < lower) | (proposed > upper), axis=1)
        proposed[invalid] = updated[invalid]
        return proposed
    if policy == "reflect":
        period = 2.0 * upper
        folded = np.mod(proposed, period)
        return np.where(folded <= upper, folded, period - folded)
    raise ValueError("current_boundary_policy must be clip, reflect, wrap, or reject_update")
