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
