# Paired regret differences — pilot1

Mean regret(engine) − regret(reference) on common consequential cards; 95 % percentile bootstrap CI (10 000 resamples, seed 0). Negative = engine better.

## vs jev

| engine | n | Δ mean regret | 95 % CI | by scenario |
|---|---:|---:|---|---|
| deepseek-lp | 170 | -3.33 | [-8.85, +1.90] | A: -4.68, B: -2.72 |
| deepseek-think | 170 | -1.88 | [-8.87, +4.99] | A: -3.42, B: -1.18 |
| fixed-milp | 170 | -0.71 | [-7.02, +5.55] | A: -2.04, B: -0.11 |
| rule-slack | 170 | +0.62 | [-4.76, +5.71] | A: -0.21, B: +0.99 |
| deepseek-fast-raw | 170 | +0.99 | [-4.53, +6.71] | A: +2.15, B: +0.47 |
| jev-raw | 170 | +1.98 | [-3.02, +6.87] | A: +6.04, B: +0.14 |
| dla-default | 170 | +2.05 | [-1.51, +6.06] | A: +2.55, B: +1.83 |
| random | 170 | +2.81 | [-3.29, +8.79] | A: +4.23, B: +2.17 |
| deepseek-fast | 170 | +4.61 | [-0.99, +10.46] | A: +5.92, B: +4.01 |

## vs dla-default

| engine | n | Δ mean regret | 95 % CI | by scenario |
|---|---:|---:|---|---|
| deepseek-lp | 170 | -5.38 | [-10.97, -0.34] | A: -7.23, B: -4.55 |
| deepseek-think | 170 | -3.93 | [-11.41, +3.21] | A: -5.96, B: -3.01 |
| fixed-milp | 170 | -2.76 | [-8.62, +3.16] | A: -4.58, B: -1.94 |
| jev | 170 | -2.05 | [-6.06, +1.51] | A: -2.55, B: -1.83 |
| rule-slack | 170 | -1.44 | [-6.29, +2.83] | A: -2.75, B: -0.84 |
| deepseek-fast-raw | 170 | -1.06 | [-5.79, +3.61] | A: -0.40, B: -1.36 |
| jev-raw | 170 | -0.08 | [-5.81, +5.26] | A: +3.49, B: -1.69 |
| random | 170 | +0.76 | [-5.41, +6.80] | A: +1.68, B: +0.34 |
| deepseek-fast | 170 | +2.55 | [-2.75, +7.88] | A: +3.38, B: +2.18 |

## vs deepseek-lp

| engine | n | Δ mean regret | 95 % CI | by scenario |
|---|---:|---:|---|---|
| deepseek-think | 170 | +1.45 | [-3.97, +6.79] | A: +1.26, B: +1.54 |
| fixed-milp | 170 | +2.62 | [-4.04, +9.51] | A: +2.64, B: +2.61 |
| jev | 170 | +3.33 | [-1.90, +8.85] | A: +4.68, B: +2.72 |
| rule-slack | 170 | +3.95 | [-1.20, +9.09] | A: +4.47, B: +3.71 |
| deepseek-fast-raw | 170 | +4.32 | [-2.28, +11.26] | A: +6.83, B: +3.19 |
| jev-raw | 170 | +5.31 | [-0.76, +11.42] | A: +10.72, B: +2.85 |
| dla-default | 170 | +5.38 | [+0.34, +10.97] | A: +7.23, B: +4.55 |
| random | 170 | +6.14 | [-0.02, +12.47] | A: +8.91, B: +4.89 |
| deepseek-fast | 170 | +7.94 | [+1.12, +15.15] | A: +10.60, B: +6.73 |
