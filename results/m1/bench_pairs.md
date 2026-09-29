# Paired regret differences — m1

Mean regret(engine) − regret(reference) on common consequential cards; 95 % percentile bootstrap CI (10 000 resamples, seed 0). Negative = engine better.

## vs jev

| engine | n | Δ mean regret | 95 % CI | by scenario |
|---|---:|---:|---|---|
| deepseek-lp | 112 | -7.59 | [-19.40, +3.13] | M: -7.59 |
| deepseek-think | 112 | -4.97 | [-15.44, +4.80] | M: -4.97 |
| jev-gate-lp | 112 | -0.53 | [-1.68, +0.18] | M: -0.53 |
| jev-ril | 112 | -0.49 | [-1.73, +0.31] | M: -0.49 |
| rule-ril420 | 112 | -0.04 | [-0.49, +0.40] | M: -0.04 |
| deepseek-fast-ril | 112 | +0.21 | [-0.39, +0.78] | M: +0.21 |
| dla-default | 112 | +0.53 | [-2.79, +5.73] | M: +0.53 |
| deepseek-fast | 112 | +0.73 | [-0.19, +1.82] | M: +0.73 |
| deepseek-think-ril | 112 | +4.51 | [-0.78, +13.96] | M: +4.51 |
| random | 112 | +5.36 | [-5.88, +20.80] | M: +5.36 |
| rule-slack | 112 | +5.75 | [-5.60, +21.29] | M: +5.75 |
| fixed-milp | 112 | +13.57 | [+1.11, +30.99] | M: +13.57 |

## vs dla-default

| engine | n | Δ mean regret | 95 % CI | by scenario |
|---|---:|---:|---|---|
| deepseek-lp | 112 | -8.12 | [-20.58, +3.31] | M: -8.12 |
| deepseek-think | 112 | -5.50 | [-16.63, +4.84] | M: -5.50 |
| jev-gate-lp | 112 | -1.05 | [-6.30, +2.40] | M: -1.05 |
| jev-ril | 112 | -1.02 | [-6.29, +2.54] | M: -1.02 |
| rule-ril420 | 112 | -0.57 | [-5.77, +2.72] | M: -0.57 |
| jev | 112 | -0.53 | [-5.73, +2.79] | M: -0.53 |
| deepseek-fast-ril | 112 | -0.31 | [-5.57, +3.05] | M: -0.31 |
| deepseek-fast | 112 | +0.21 | [-5.04, +3.71] | M: +0.21 |
| deepseek-think-ril | 112 | +3.98 | [-4.20, +14.98] | M: +3.98 |
| random | 112 | +4.83 | [-7.47, +20.94] | M: +4.83 |
| rule-slack | 112 | +5.22 | [-7.28, +21.38] | M: +5.22 |
| fixed-milp | 112 | +13.04 | [-0.36, +30.63] | M: +13.04 |

## vs deepseek-lp

| engine | n | Δ mean regret | 95 % CI | by scenario |
|---|---:|---:|---|---|
| deepseek-think | 112 | +2.62 | [-1.54, +8.77] | M: +2.62 |
| jev-gate-lp | 112 | +7.06 | [-3.57, +18.85] | M: +7.06 |
| jev-ril | 112 | +7.10 | [-3.54, +18.88] | M: +7.10 |
| rule-ril420 | 112 | +7.54 | [-3.18, +19.39] | M: +7.54 |
| jev | 112 | +7.59 | [-3.13, +19.40] | M: +7.59 |
| deepseek-fast-ril | 112 | +7.80 | [-2.94, +19.54] | M: +7.80 |
| dla-default | 112 | +8.12 | [-3.31, +20.58] | M: +8.12 |
| deepseek-fast | 112 | +8.32 | [-2.47, +19.99] | M: +8.32 |
| deepseek-think-ril | 112 | +12.10 | [-0.72, +27.10] | M: +12.10 |
| random | 112 | +12.95 | [-1.37, +30.78] | M: +12.95 |
| rule-slack | 112 | +13.34 | [+1.10, +29.50] | M: +13.34 |
| fixed-milp | 112 | +21.16 | [+4.27, +40.98] | M: +21.16 |
