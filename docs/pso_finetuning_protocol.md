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
- Fine-tuning evaluates the optimizer at the initial network state only (one
  clustering/optimization round). FND is not a tuning criterion.
- Network lifetime and FND must be validated in a separate later experiment.

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
final best J. Statistical indistinguishability is assessed with a deterministic
paired bootstrap 95% confidence interval (10,000 resamples) for the difference
from the minimum-mean candidate. Among statistically indistinguishable candidates,
prefer lower population standard deviation, then lower runtime, then retain the
user-specified candidate order.

1. Dimension probe: record the actual EULC candidate-vector dimension D and use
   `ceil(10 + 2*sqrt(mean(D)))`, clamped to 20-30, as the baseline N.
2. Inertia schedule: with baseline `max_iter=50`, baseline N and c1=c2=2.0,
   compare fixed 0.9 plus linear schedules 0.9->0.4, 0.9->0.2 and 1.0->0.4.
3. Acceleration coefficients: retain the selected inertia strategy and compare
   (2.0,2.0), (2.5,1.5), (1.5,2.5), and (1.49445,1.49445).
4. Plateau and N: plateau means relative best-J improvement below 0.1% for 20
   consecutive iterations. Add 25% to the detected median plateau iteration.
   Also report sensitivity at 0.05%/10 and 0.5%/30. Compare N=20,30,50 at the
   proposed iteration count. If J improvement is below 0.1%, prefer lower runtime.
5. Velocity limit: measure mean per-dimension particle standard deviation around
   the centroid. Because priorities are dimensionless [0,1], scale this value by
   the physical 100x100x100 m diagonal for the required diagnostic. Only compare
   Vmax=1.0,0.1,0.2 when plateau is detected and mean diversity over the final 20
   iterations exceeds 8.660254 m (5% of the 173.205081 m diagonal).
6. Final comparison: compare the tuned configuration with the project default
   (fixed w = 0.9, c1 = c2 = 2.0, N = 30) on 30 independent paired seeds.

The default comparator uses fixed w=0.9, c1=c2=2.0, N=30, max_iter=50 and
Vmax=1.0. Final comparison uses 30 paired independent seeds.

## Required outputs

For every run save: scenario identity, seed, observed particle dimension,
population, iterations, inertia schedule/start/end, c1, c2, Vmax, initial and
final best J, improvement, runtime, and convergence/diversity history. Lifetime
metrics are deliberately excluded from parameter selection.

For every scenario save:

- `trials.csv`
- `convergence.csv`
- `step_winners.csv`
- `all_step_rankings.csv`
- `plateau_sensitivity.csv`
- `diversity_diagnostic.json`
- `final_convergence_summary.csv`
- `best_config.yaml`
- `selection_explanation.md`
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

## Reusing this protocol for another scenario matrix

Treat this file as the mandatory checklist whenever a future request says to
fine-tune PSO. Do not silently reuse the winning configuration from another
network state. Create one independent tuning result for every combination of
deployment size, node count/density, distribution, packet size, initial energy,
and transmission range.

To define a new matrix:

1. Copy `configs/tuning/local_pso_uniform_matrix.yaml` to a new versioned config.
2. Change only the scenario dimensions explicitly requested by the experiment:
   deployment size, density/node mapping, distribution, packet-energy cases,
   and transmission range.
3. Keep the sequential tuning rules, seed policy, selection metric, plateau
   thresholds, diversity gate, baseline, and output schema unchanged unless the
   user explicitly approves a new protocol version.
4. Give every scenario a stable ID containing density, node count, packet size,
   and initial energy. Add further environment fields to the ID when they vary.
5. Use a separate output directory per scenario. Never pool raw J across
   scenarios because their network states and normalization denominators differ.
6. Validate a saved result before skipping it. It is reusable only when its
   schema/method version, scenario, objective, seed policy, and Git commit are
   compatible. Preserve incompatible historical results, but do not use them to
   select the new winner.
7. After optimizer tuning, run a separate lifetime experiment using the saved
   `best_config.yaml`; only that later experiment may report FND/HND/LND claims.

## Kaggle notebook template

Use one notebook and one checkpoint Dataset per independent density or other
manageable scenario group. Attach that Dataset to the notebook before enabling
scheduled, non-interactive execution. Pin the Git commit so a resumed campaign
cannot change code midway.

```bash
%%bash
set -euxo pipefail

REPO=/kaggle/working/fine_tune
OUTPUT=/kaggle/working/<OUTPUT_NAME>
COMMIT=<VERIFIED_GIT_COMMIT>

if [ -d "$REPO/.git" ]; then
    git -C "$REPO" fetch origin
else
    git clone https://github.com/nvm1cz/fine_tune.git "$REPO"
fi

git -C "$REPO" checkout --detach "$COMMIT"
cd "$REPO"
python -m pip install -q -r requirements.txt

python -u scripts/run_kaggle_pso_uniform_matrix_tuning.py \
    --density <sparse|medium|dense> \
    --config <MATRIX_CONFIG_PATH> \
    --output-dir "$OUTPUT" \
    --checkpoint-dataset <KAGGLE_USERNAME/DATASET_SLUG> \
    --max-wall-time-seconds 39600 \
    --execute
```

Recommended Kaggle settings are: CPU, no accelerator, Internet enabled while
cloning/installing, and Files-only persistence. A daily schedule may resume the
checkpoint. Disable the schedule after the manifest reports all scenarios
complete to avoid unnecessary Dataset versions.

## Completion audit

Before accepting a scenario as tuned, verify all of the following:

- `result.json` has the expected method/schema version and exact scenario;
- all intermediate candidates have at least 10 independent paired seeds;
- the final baseline-versus-tuned comparison has 30 paired seeds;
- rankings use only mean best J for the primary decision;
- every ranking contains mean best J, standard deviation, and runtime;
- plateau sensitivity contains all three required threshold/patience settings;
- diversity contains a measured numeric value and states whether Vmax tuning ran;
- `best_config.yaml` agrees with the selected rows in the CSV files;
- convergence PNG and PDF exist;
- the report explicitly says lifetime validation is still required.
