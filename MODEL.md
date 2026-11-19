# Model: cohort forecast of GTA VI trophy earn rates

## The quantity

Each case asks for every PS5 trophy's earn rate on Exophase at a target
instant: the percent of the players Exophase tracks for the game who have
earned that trophy. The program (`main.py`, Python 3 standard library
only) reads the case's history of page reads and writes five quantiles
(5%, 25%, 50%, 75%, 95%) per trophy.

## Why a trend line is the wrong shape

On launch night the page is flat — 638 players tracked, every rate the
same — and then it refreshes to 3,868 players with every rate lower.
Two forces move a rate, in opposite directions:

- **Dilution:** new players join tracked with zero trophies, pulling
  every rate down.
- **Progression:** the players already tracked keep playing and earning,
  pushing every rate up.

A trend fitted to the rates alone cannot see that tug-of-war, and it
cannot know that a rate can fall while every player is progressing.
So the model tracks the two forces separately.

## The model

**Players tracked, N(t).** Inside the case's history the reads are
trusted (log-linear interpolation between them). Beyond the last read,
a saturating adoption curve (Weibull CDF) fitted to the reads, blended
in log space with a power-law fit, extrapolates N to the target.

**Founders.** The players already tracked at the first read joined
before release — the first read already shows trophies earned. They
form a founder cohort spread over [-a0, 0), where a0 (0, 3, 6 or 12
hours) is the head start whose fits explain the reads best.

**Earning.** A player who joined at time s and is a hours old has
earned trophy j with probability

    F_j(a) = sigmoid((ln a - ln m_j) / w)

a log-logistic time-to-earn curve with a per-trophy median m_j and a
shared spread w. A trophy's rate at time t is the cohort-weighted mean
of F_j over every cohort that has joined by t. New cohorts (F ~ 0) pull
the rate down; ageing cohorts push it up.

**Fitting.** m_j is fitted per trophy by least squares on the logit
scale against the case's reads (golden-section search on ln m_j over a
small grid of w), with a small penalty pulling ln m_j toward a prior
built from the trophy's tier and description (story chapters early,
long grinds late). With a flat history the fit still pins the level —
the fitted m reproduces the observed rate — and the penalty breaks the
tie between the (w, m) pairs that match it. With two days of real
reads the data takes over. One trophy is special: the "Gone Fishing"
trophy sits at ~100% in every read, so it is held at its last rate.

**Quantiles.** The median is the cohort model's rate at the target.
The spread on the logit scale combines a random-walk term growing with
the horizon, a model-uncertainty term, the trophy's in-sample residual
and a floor for Exophase's discrete refresh jumps, so the 90% band is
honest about a 6.5-day launch-week forecast.

## Validation

- On the kit's practice cases the program is valid, deterministic
  (byte-identical over repeated runs) and fast (~0.5 s).
- On a synthetic final-like case (60 h of reads, target 4.5 days
  later, ground truth from an independent cohort process with
  different parameters), it scores a pinball loss of ~0.39 against
  ~0.68 for the house baseline — the cohort structure, not the trend,
  is what earns the gap.
- Degenerate inputs (one read, missing rates, no history) all produce
  valid forecasts.
