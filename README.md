# UWSN Kaggle Tuning

Minimal repository for reproducible hyperparameter tuning of:

- EULC + PSO
- EULC + GA
- EULC + AC-ACO

The three optimizers share the same EULC candidate generation, channel, energy,
delay, routing, objective and experiment inputs.

## Install

```bash
pip install -r requirements.txt
```

## Test

```bash
python -m unittest discover -s tests
```

## Generate scenarios

```bash
python scripts/generate_experiment_inputs.py \
  --experiment-set configs/experiment_sets/benchmark.yaml \
  --output-dir configs/generated/benchmark \
  --clean
```

## Run one case

```bash
python scripts/run_experiment_case.py \
  --config configs/generated/benchmark/cases/case_000001.yaml \
  --algorithm configs/algorithms/eulc_pso.yaml
```

Generated cases and result directories are excluded from Git.

## Tune on Kaggle

The tuner uses the same benchmark scenarios and scoring policy for PSO, GA and
AC-ACO. It applies successive halving: many configurations are screened cheaply,
then only the strongest configurations reach the 200-round and 1,000-round
stages.

Preview the workload without running simulations:

```bash
python scripts/run_kaggle_tuning.py
```

Run a small end-to-end check:

```bash
python scripts/run_kaggle_tuning.py \
  --smoke \
  --execute \
  --workers 1
```

Run the full search on Kaggle:

```bash
python scripts/run_kaggle_tuning.py \
  --execute \
  --workers 4
```

To run one optimizer per Kaggle session, give each optimizer its own output
directory:

```bash
python scripts/run_kaggle_tuning.py --algorithm eulc_pso \
  --output-dir /kaggle/working/uwsn_tuning_pso --execute --workers 4
python scripts/run_kaggle_tuning.py --algorithm eulc_ga \
  --output-dir /kaggle/working/uwsn_tuning_ga --execute --workers 4
python scripts/run_kaggle_tuning.py --algorithm eulc_ac_aco \
  --output-dir /kaggle/working/uwsn_tuning_ac_aco --execute --workers 4
```

For scheduled runs, use a separate private Kaggle Dataset for each optimizer.
The runner restores its latest checkpoint before tuning and creates a new
Dataset Version immediately after every completed simulation:

```bash
python scripts/run_kaggle_tuning.py --algorithm eulc_pso \
  --output-dir /kaggle/working/uwsn_tuning_pso \
  --checkpoint-dataset nguyenvuminh/uwsn-checkpoint-pso \
  --execute --workers 4

python scripts/run_kaggle_tuning.py --algorithm eulc_ga \
  --output-dir /kaggle/working/uwsn_tuning_ga \
  --checkpoint-dataset nguyenvuminh/uwsn-checkpoint-ga \
  --execute --workers 4

python scripts/run_kaggle_tuning.py --algorithm eulc_ac_aco \
  --output-dir /kaggle/working/uwsn_tuning_ac_aco \
  --checkpoint-dataset nguyenvuminh/uwsn-checkpoint-ac-aco \
  --execute --workers 4
```

If remote upload fails after three attempts, the runner stops instead of
continuing without a recoverable checkpoint. Kaggle scheduled runs start at
their configured time; they do not automatically start at the exact instant a
previous 12-hour session ends.

The checkpoint backend sets `DISABLE_KAGGLE_CACHE=true` before importing
KaggleHub. This is required because non-interactive scheduled sessions cannot
attach new Dataset inputs; the authenticated HTTP resolver downloads the latest
checkpoint Dataset version instead.

Results are written to `/kaggle/working/uwsn_tuning`. Re-running the same
command resumes completed trials from `trials.jsonl`. The main artifacts are:

- `best_config_pso.yaml`, `best_config_ga.yaml`, `best_config_ac_aco.yaml`;
- `best_summary.json`;
- `rankings_<stage>.csv` and `rankings_all_stages.csv`;
- `manifest.json` and `trials.jsonl`.
- `progress.json` for the current machine-readable counter;
- `progress.log` for the append-only human-readable progress history.

Each completed simulation is flushed to `trials.jsonl` before the next progress
update is published. To inspect a running Kaggle job:

```bash
cat /kaggle/working/uwsn_tuning_pso/progress.json
tail -n 20 /kaggle/working/uwsn_tuning_pso/progress.log
```

The search space, stage budgets, seeds and scenario filters are defined in
`configs/tuning/kaggle_large.yaml`. Raw per-round objective values are not
averaged across different network states; configurations are ranked using
run-level lifetime, residual-energy AUC, delivery, delay and runtime metrics.
