# makefun-gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rizkydncrypto-tech/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rizkydncrypto-tech/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A cohort model that forecasts each GTA VI (PS5) trophy's Exophase earn rate in week one: founders with a pre-release head start, saturating growth in players tracked, and a log-logistic time-to-earn curve per trophy, fitted to the case's own reads. The model separates the two forces the launch-night page shows — new players diluting the rates while the players already tracked keep progressing — instead of fitting a trend to the rates alone. See MODEL.md for the details.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3, standard library only) reads every `*.in` case in the input directory and writes one `*.out` forecast per case. No network, no dependencies, deterministic.

---

<sub>Entered on [make.fun](https://make.fun/?ref=dnpvszdc&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
