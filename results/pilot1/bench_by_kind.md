# Bench by decision kind — pilot1

## depart

| engine | n | agree | regret | Σregret | non-default |
|---|---:|---:|---:|---:|---:|
| deepseek-think | 50 | 0.440 | 11.50 | 575 | 0.80 |
| fixed-milp | 54 | 0.389 | 13.30 | 718 | 1.00 |
| jev | 54 | 0.611 | 15.15 | 818 | 0.04 |
| rule-slack | 54 | 0.648 | 15.06 | 813 | 0.04 |
| dla-default | 54 | 0.611 | 17.78 | 960 | 0.00 |
| random | 54 | 0.444 | 20.37 | 1100 | 0.54 |
| deepseek-fast | 54 | 0.389 | 13.26 | 716 | 0.93 |

## meet

| engine | n | agree | regret | Σregret | non-default |
|---|---:|---:|---:|---:|---:|
| deepseek-think | 112 | 0.509 | 12.11 | 1356 | 0.64 |
| fixed-milp | 116 | 0.741 | 12.58 | 1459 | 0.29 |
| jev | 116 | 0.672 | 12.76 | 1480 | 0.22 |
| rule-slack | 116 | 0.560 | 13.71 | 1590 | 0.35 |
| dla-default | 116 | 0.759 | 14.54 | 1687 | 0.00 |
| random | 116 | 0.500 | 14.45 | 1676 | 0.53 |
| deepseek-fast | 116 | 0.534 | 20.39 | 2365 | 0.24 |
