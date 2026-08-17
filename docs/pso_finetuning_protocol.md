# PSO Fine-Tuning Protocol for the UWSN Simulator

This file is the project instruction for future PSO fine-tuning. Read it before
starting any new PSO tuning campaign.

## Fixed experimental scope

- Deployment distribution: uniform only.
- Deployment volume: 100 m x 100 m x 100 m.
- Transmission range: 150 m.
- Densities: sparse = 50 nodes, medium = 100 nodes, dense = 150 nodes.
- Packet sizes: 4000 and 6400 bits.
- Initial node energies: 0.5 and 1.0 J.
- Each density therefore has four packet-energy scenarios.
- A simulation run stops immediately after the first node death (FND).
- `max_simulation_rounds = 500` is only a safety/censoring ceiling.

Do not compare or pool raw objective J values across different network states.
Each scenario is tuned and reported independently.

## Objective and particle dimension

The optimizer minimizes the project's shared normalized multiplicative
energy-delay objective. A particle is a continuous priority vector over the
current EULC candidate set. Its dimension D is therefore the number of eligible
candidate nodes at the optimizer refresh, not a fixed global constant. The
decoder selects the required number of cluster heads from this priority vector.
Record the observed D for every seed.

## Sequential OFAT procedure

All comparisons use paired deterministic seeds. Screening uses seeds 0-9 and
the final comparison uses independent seeds 10-39. Select by the lowest mean
final best J; if means tie, select the lowest population standard deviation; if both
tie, retain the user-specified candidate order.

1. Population pre-screen: compare N = 20 and N = 30 with fixed w = 0.9,
   c1 = c2 = 2.0, and the initial `max_iter = 200`.
2. Inertia schedule: with the selected N and c1 = c2 = 2.0, compare linear
   schedules 0.9 -> 0.4, 0.9 -> 0.2, and 1.0 -> 0.4.
3. Acceleration coefficients: retain the selected inertia strategy and compare
   (2.0, 2.0), (2.5, 1.5), (1.5, 2.5). Compare the Clerc-Kennedy pair
   (1.49445, 1.49445) only with fixed w = 0.729.
4. Population/iteration retest: compare the Cartesian grid N in {20, 30} and
   `max_iter` in {100, 150, 200}, using the selected w, c1, and c2. Inspect the
   saved convergence curves when interpreting whether the winning iteration
   budget is already on a plateau. The project does not invent a numerical
   plateau threshold; a future automatic plateau rule remains TODO until the
   user confirms its tolerance and patience.
5. Velocity limit: compare Vmax = 1.0, 0.2, and 0.1 in the normalized [0,1]
   priority space. Vmax 0.1 and 0.2 correspond to 10% and 20% of the variable
   range. Do not claim Vmax was necessary without convergence evidence.
6. Final comparison: compare the tuned configuration with the project default
   (fixed w = 0.9, c1 = c2 = 2.0, N = 30) on 30 independent paired seeds.

The default comparator uses fixed w = 0.9, c1 = c2 = 2.0, N = 30, and
`max_iter = 200`, exactly as requested.

## Required outputs

For every run save: scenario identity, seed, observed particle dimension,
population, iterations, inertia schedule/start/end, c1, c2, Vmax, initial and
final best J, improvement, runtime, FND or censoring state, rounds executed,
alive nodes and residual energy at stop, packet-delivery ratio, and convergence
history.

For every scenario save:

- `trials.csv`
- `convergence.csv`
- `phase_winners.csv`
- `all_config_rankings.csv`
- `result.json`
- default-versus-tuned convergence plots in PNG and PDF

## Reproduction command

```bash
python3 scripts/run_local_pso_uniform_matrix_tuning.py \
  --config configs/tuning/local_pso_uniform_matrix.yaml \
  --output-dir results/local_pso_uniform_matrix
```

The matrix runner skips a scenario when its `result.json` already exists. This
allows a stopped campaign to resume without rerunning completed scenarios.

For scheduled Kaggle execution, use `run_kaggle_pso_uniform_matrix_tuning.py`.
Use one notebook and one attached checkpoint Dataset per density. The script
restores the latest ZIP snapshot, skips completed scenarios, publishes after
each scenario, and exits cleanly before its wall-time budget when possible.
