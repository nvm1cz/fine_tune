# Energy-Triggered Re-clustering for EULC + PSO

## Source and project adaptation

Yi et al. (Sensors 2023, 23, 5466; DOI 10.3390/s23125466) describe a cluster
maintenance rule in which a cluster is reorganized when its cluster-head (CH)
residual energy falls below the mean residual energy of that cluster. The paper
implements NUC-EB, not EULC + PSO.

The project therefore uses the same trigger but preserves its joint CH-routing
architecture: each cluster is checked independently, and if at least one cluster
triggers, the shared EULC + PSO optimizer rebuilds the global CH set and route
plan. A strictly local PSO replacement is not used because the current objective
jointly evaluates assignments, forwarding load, and inter-cluster routing.

## Configuration

```yaml
protocol:
  recluster_trigger_mode: cluster_energy_mean
```

Supported modes:

- `periodic`: historical behavior controlled by `recluster_interval_rounds`.
- `cluster_energy_mean`: no periodic refresh; refresh when any alive CH is below
  the mean energy of the alive nodes in its cluster. Dead/invalid CH and broken
  route safety triggers remain active.
- `hybrid`: periodic and energy-mean triggers are both enabled.

The historical default remains `periodic`, so existing campaign checkpoints and
results are not silently changed.

## Smoke benchmark

Scenario: sparse, 50 nodes, uniform, 100 x 100 x 100 m, 4000 bits, 0.5 J,
transmission range 150 m, three paired seeds, up to 50 rounds.

| Mode | Mean optimizer refreshes | Mean runtime (s) | Mean FND | Mean HND | Mean LND |
|---|---:|---:|---:|---:|---:|
| periodic each round | 44.33 | 8.26 | 13.67 | 26.00 | 44.33 |
| cluster energy mean | 26.00 | 6.17 | 14.00 | 25.67 | 46.00 |

The trigger reduced optimizer refreshes by about 41.4% and runtime by about
25.3% in this small smoke test. This is preliminary evidence only; a 30-seed
paired lifetime validation is required before making a lifetime claim.
