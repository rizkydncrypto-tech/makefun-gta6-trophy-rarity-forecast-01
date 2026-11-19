"""Forecast GTA VI (PS5) trophy earn rates on Exophase at a case's target.

Model: a cohort model of how a tracked-player population grows and ages.

Why this shape, and not a per-trophy trend line: on launch night the page
is flat (same rates, same 638 players tracked) and then refreshes to 3,868
players with every rate lower -- new players dilute the rates while the
players who were already tracked keep progressing. A trend fitted to the
rates alone cannot see that tug-of-war. So this program models the two
forces separately and combines them:

  1. Players tracked, N(t): inside the case's history the reads
     themselves are trusted (log-linear interpolation between them);
     beyond the last read a saturating adoption curve (Weibull CDF)
     fitted to the reads, blended in log space with a power-law fit,
     says how many players are tracked at the target -- and how young
     that population is. The players already tracked at the first
     read are founders: they joined before release, spread over
     [-a0, 0), and a0 (0, 3, 6 or 12 hours) is the head start whose
     fits explain the reads best. That matters because the first read
     already shows trophies earned: the founders were playing before
     the release timestamp.
  2. Earning, per trophy: a player who joined s hours after release and
     is a hours old has earned trophy j with probability
     F_j(a) = sigmoid((ln a - ln m_j) / w), a log-logistic "time to
     earn" curve with a per-trophy median m_j and a shared spread w.
     A trophy's rate at time t is the cohort-weighted mean of F_j
     over every cohort that has joined by t. New cohorts (F ~ 0) pull
     the rate down; ageing cohorts push it up -- exactly the dilution
     versus progression the page shows.
  3. m_j is fitted per trophy by least squares on the logit scale
     against the case's reads (golden-section search on ln m_j, over a
     small grid of shared w), with a small penalty pulling ln m_j
     toward a prior built from the trophy's tier and description
     (story chapters early, grinds late). With a flat history the fit
     still pins the level -- the fitted m reproduces the observed rate
     -- and the penalty breaks the tie between the (w, m) pairs that
     match it; with two days of real reads the data takes over.

Quantiles: the median is the cohort model's rate at the target. The
spread on the logit scale combines a random-walk term that grows with
the horizon, a model-uncertainty term, the trophy's in-sample residual
and a floor for Exophase's discrete refresh jumps, so the 90% band is
honest about a 6.5-day launch-week forecast.

Standard library only, deterministic (no randomness, no dict-order or
clock dependence), reads every *.in in the input directory and writes
one *.out per case. Run locally with:
    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Fitting budgets, kept small so a run stays far inside its 8 seconds.
MAX_FIT_READS = 12          # reads used to fit the earning curves
MAX_N_POINTS = 60           # reads used to fit the players curve
W_GRID = (0.5, 0.8, 1.2, 1.8, 2.6)
LN_M_LO = math.log(0.3)     # median time-to-earn: 0.3 h ..
LN_M_HI = math.log(5000.0)  # .. 5000 h


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def clamp_rate(pct):
    return clamp(pct, 0.01, 99.99)


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x)) if x < 700 else 1.0
    e = math.exp(x) if x > -700 else 0.0
    return e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, as the house baseline does."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(value):
    """ISO 8601 date or datetime -> aware datetime (UTC). None if unparsable."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    try:
        if len(text) == 10:  # a bare date: start of that day, UTC
            return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def hours_between(start, end):
    return (end - start).total_seconds() / 3600.0


# ---------------------------------------------------------------- N(t)

def fit_players_curve(points):
    """Fit N(t) from (t_hours, N) reads.

    Returns a function N(t) for t >= 0: a Weibull adoption curve fitted
    on log N, blended 50/50 in log space with a power-law fit. With too
    little variation the Weibull falls back to a launch-week prior
    (fast early growth, saturating), which is what a flat launch-night
    history needs: the page is stale, the true tracked count is growing.
    """
    pts = [(t, n) for (t, n) in points if n and n > 0 and t >= 0]
    if not pts:
        return lambda t: 1000.0
    if len(pts) > MAX_N_POINTS:
        idx = subsample_indices(len(pts), MAX_N_POINTS)
        pts = [pts[i] for i in idx]
    n_last = pts[-1][1]
    distinct_n = len({n for _, n in pts})
    t_last = pts[-1][0]

    def weibull(t, n_inf, tau, k):
        if t <= 0:
            return 0.0
        return n_inf * (1.0 - math.exp(-((t / tau) ** k)))

    if distinct_n >= 3 and len(pts) >= 3:
        n_max = max(n for _, n in pts)
        best = None  # (sse, n_inf, tau, k)
        taus = [2.0 * (200.0 ** (i / 24.0)) for i in range(25)]  # 2..400 h
        for k in (0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2):
            for mult in (1.1, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0):
                n_inf = n_max * mult
                for tau in taus:
                    sse = 0.0
                    for t, n in pts:
                        pred = weibull(t, n_inf, tau, k)
                        if pred <= 0:
                            sse += 100.0
                            continue
                        d = math.log(n) - math.log(pred)
                        sse += d * d
                    if best is None or sse < best[0]:
                        best = (sse, n_inf, tau, k)
        _, n_inf, tau, k = best
        # Power-law fit: ln N = a + b ln(1 + t), by least squares.
        xs = [math.log1p(t) for t, _ in pts]
        ys = [math.log(n) for _, n in pts]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        b = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
             if sxx > 1e-12 else 0.0)
        a = my - b * mx

        def curve(t):
            w_pred = weibull(t, n_inf, tau, k)
            p_pred = math.exp(a + b * math.log1p(t)) if t >= 0 else 0.0
            if w_pred <= 0:
                return max(p_pred, 1.0)
            if p_pred <= 0:
                return max(w_pred, 1.0)
            return math.exp(0.5 * (math.log(w_pred) + math.log(p_pred)))
    else:
        # Flat history: prior curve, N_inf 25x the last count, tau 96 h.
        n_inf, tau, k = n_last * 25.0, 96.0, 0.7

        def curve(t):
            return max(weibull(t, n_inf, tau, k), 1.0)

    n_at_last = curve(t_last)
    scale = n_last / n_at_last if n_at_last > 0 else 1.0

    def anchored(t):
        return max(curve(t) * scale, 1.0)

    return anchored


def cohort_edges(t_end):
    """Bin edges for cohorts, log-spaced: fine where players are young
    (their earn probability moves fastest there on the log-age scale)
    and coarse later. Every edge is a join time in hours since release."""
    edges = [0.0]
    e = 0.25
    while e < t_end:
        edges.append(e)
        e *= 1.6
    edges.append(t_end)
    return edges


def cohort_age(t, s0, s1):
    """Representative age at time t of the cohort that joined in [s0, s1):
    the geometric mean of the ages its members span, since F_j is a
    function of log age."""
    lo = max(t - s1, 0.05)
    hi = max(t - s0, 0.05)
    return math.sqrt(lo * hi)


# ------------------------------------------------------- prior for m_j

def prior_ln_m(trophy):
    """Prior median time-to-earn (hours) from tier and description."""
    tier = (trophy.get("tier") or "bronze").lower()
    base = {"platinum": 1500.0, "gold": 350.0, "silver": 150.0,
            "bronze": 70.0}.get(tier, 70.0)
    text = ((trophy.get("name") or "") + " " +
            (trophy.get("description") or "")).lower()
    m = base
    # Story order: chapters and the finale, earliest first.
    if "chapter 1" in text:
        m = 25.0
    elif "chapter 2" in text:
        m = 45.0
    elif "chapter 3" in text:
        m = 70.0
    elif "chapter 4" in text:
        m = 100.0
    elif "final mission" in text:
        m = 140.0
    # Long grinds take far longer than their tier suggests.
    grindy = (("100% completion" in text, 1500.0),
              ("250 stunts" in text, 900.0),
              ("collectible" in text, 500.0),
              ("50 challenges" in text, 400.0),
              ("20 different races" in text, 250.0),
              ("every location" in text, 300.0),
              ("every spot" in text, 300.0),
              ("each gang" in text, 250.0))
    for hit, value in grindy:
        if hit:
            m = max(m, value)
    # Quick one-offs are fast whatever their tier.
    if any(k in text for k in ("arcade game", "cocktail", "meditate",
                               "fishing challenge", "discover 100 vehicles")):
        m = min(m, 45.0)
    return math.log(clamp(m, 5.0, 2000.0))


# ------------------------------------------------------------ the case

def subsample_indices(n, k):
    """k indices spread over range(n), always including 0 and n-1."""
    if n <= k:
        return list(range(n))
    if k <= 1:
        return [n - 1]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_min = float(case.get("horizon_minutes") or 0)
    d_days = horizon_min / 1440.0
    target = parse_time(case.get("target"))
    release = parse_time((case.get("game") or {}).get("release_date"))

    reads = []  # (t_hours, players, rates)
    for h in history:
        at = parse_time(h.get("at")) or parse_time(h.get("slot"))
        if at is None:
            continue
        origin = release or at
        t = max(hours_between(origin, at), 0.05)  # reads just before
        # release (the first recorder started at 23:50) count as t ~ 0.
        rates = h.get("rates") or []
        reads.append((t, h.get("players_tracked"), rates))
    reads.sort(key=lambda r: r[0])
    if not reads:
        # No history at all: pure-prior forecast at the target age.
        t_target = hours_between(release, target) if (release and target) else 168.0
        rows = []
        for tr in trophies:
            med = prior_median_rate(t_target, tr)
            rows.append((tr["id"], med, 0.6))
        return build_rows(rows, d_days)

    if release is None:
        # No release date: measure time from the first read instead, so
        # every age below is hours since the first read.
        shift = reads[0][0]
        t_target = reads[-1][0] - shift + horizon_min / 60.0
    else:
        shift = 0.0
        t_target = hours_between(release, target) if target else \
            reads[-1][0] + horizon_min / 60.0
    # Work in hours since the first read when there is no release date.
    t_tgt = t_target - shift

    n_curve = fit_players_curve([(t, n) for (t, n, _) in
                                 [(r[0] - shift, r[1], r[2]) for r in reads]])
    n_last = reads[-1][1] or 0
    t_first = reads[0][0] - shift
    t_last = reads[-1][0] - shift
    n_first = reads[0][1] or 0

    # Players tracked, N(t): trust the reads inside history (log-linear
    # interpolation between them) and extrapolate with the fitted curve
    # beyond the last read, anchored to it. The fitted curve alone
    # would invent growth inside a flat history; the reads say it was
    # flat.
    obs_t, obs_n = [], []
    for (t, n, _) in reads:
        tt = t - shift
        if n and n > 0 and (not obs_t or tt > obs_t[-1]):
            obs_t.append(tt)
            obs_n.append(float(n))
    curve_at_last = n_curve(t_last)
    curve_scale = (n_last / curve_at_last) if curve_at_last > 0 else 1.0

    def n_used(t):
        if t <= t_first:
            return float(n_first) if n_first else 1.0
        if t <= t_last:
            if len(obs_t) == 1:
                return float(obs_n[0])
            for i in range(len(obs_t) - 1):
                if obs_t[i] <= t <= obs_t[i + 1]:
                    span = obs_t[i + 1] - obs_t[i]
                    f = (t - obs_t[i]) / span if span > 0 else 0.0
                    return math.exp((1.0 - f) * math.log(obs_n[i]) +
                                    f * math.log(obs_n[i + 1]))
            return float(obs_n[-1])
        return max(n_curve(t) * curve_scale, 1.0)

    # Cohort bins: the founders -- the players already tracked at the
    # first read -- joined before release, spread over [-a0, 0); later
    # bins follow n_used's growth. Weights come from n_used; a bin's
    # age is the geometric mean of the ages its members span.
    t_end = max(t_tgt, max((r[0] - shift) for r in reads))
    edges = cohort_edges(t_end)

    def bins_for(a0):
        bins = []
        if a0 > 0:
            bins.append((-a0, -0.5 * a0, 0.5 * n_first))
            bins.append((-0.5 * a0, 0.0, 0.5 * n_first))
            prev = float(n_first)
        else:
            prev = n_used(0.0)
        for s0, s1 in zip(edges, edges[1:]):
            cur = n_used(s1)
            w = cur - prev
            if w > 0:
                bins.append((s0, s1, w))
            prev = cur
        return bins

    # --- fit the earning curves on a subsample of the reads --------
    fit_idx = subsample_indices(len(reads), MAX_FIT_READS)
    fit_reads = [reads[i] for i in fit_idx]
    fit_t = [r[0] - shift for r in fit_reads]

    def row_matrices(bins, times):
        """(log-age, normalised weight) rows, one per time in `times`."""
        ln_rows, w_rows = [], []
        for t in times:
            row_a, row_w, tot = [], [], 0.0
            for s0, s1, wgt in bins:
                if s0 >= t:
                    break
                age = cohort_age(t, s0, s1)
                row_a.append(math.log(age))
                row_w.append(wgt)
                tot += wgt
            if tot <= 0:
                row_a, row_w = [math.log(max(t, 0.05))], [1.0]
                tot = 1.0
            ln_rows.append(row_a)
            w_rows.append([w / tot for w in row_w])
        return ln_rows, w_rows

    def predict_row(ln_a_row, w_row, ln_m, w):
        total = 0.0
        for la, wt in zip(ln_a_row, w_row):
            total += wt * sigmoid((la - ln_m) / w)
        return clamp(total, 0.0001, 0.9999)

    def sse_for(ln_m, w, y_obs, ln_age, w_norm):
        sse = 0.0
        for r, y in enumerate(y_obs):
            if y is None:
                continue
            pred = predict_row(ln_age[r], w_norm[r], ln_m, w)
            d = math.log(pred / (1.0 - pred)) - y
            sse += d * d
        return sse

    def sse_penalised(ln_m, w, y_obs, ln_m_prior, ln_age, w_norm):
        # Least squares plus a small pull toward the prior median
        # time-to-earn. With reads that move, the data dominates; with
        # a flat history every (w, m) that reproduces the flat level
        # fits equally well, and the penalty picks the one nearest the
        # prior instead of an arbitrary corner of the grid.
        return (sse_for(ln_m, w, y_obs, ln_age, w_norm) +
                0.1 * (ln_m - ln_m_prior) ** 2)

    def golden_search_lnm(w, y_obs, ln_m_prior, ln_age, w_norm, iters=16):
        """Minimise the penalised SSE over ln m in [LN_M_LO, LN_M_HI]."""
        lo, hi = LN_M_LO, LN_M_HI
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        fc = sse_penalised(c, w, y_obs, ln_m_prior, ln_age, w_norm)
        fd = sse_penalised(d, w, y_obs, ln_m_prior, ln_age, w_norm)
        for _ in range(iters):
            if fc < fd:
                hi, d, fd = d, c, fc
                c = hi - gr * (hi - lo)
                fc = sse_penalised(c, w, y_obs, ln_m_prior, ln_age, w_norm)
            else:
                lo, c, fc = c, d, fd
                d = lo + gr * (hi - lo)
                fd = sse_penalised(d, w, y_obs, ln_m_prior, ln_age, w_norm)
        return 0.5 * (lo + hi)

    def fit_trophy(ln_age, w_norm, y_obs, ln_m_prior, w_grid, iters=16):
        """Best (ln m, w) for one trophy; returns (ln_m, w, raw SSE)."""
        best = None  # (penalised sse, ln_m, w)
        for w in w_grid:
            ln_m = golden_search_lnm(w, y_obs, ln_m_prior, ln_age, w_norm,
                                     iters)
            score = sse_penalised(ln_m, w, y_obs, ln_m_prior, ln_age, w_norm)
            if best is None or score < best[0]:
                best = (score, ln_m, w)
        _, ln_m, w = best
        return ln_m, w, sse_for(ln_m, w, y_obs, ln_age, w_norm)

    # Observed logit rates per trophy, on the fit reads.
    y_all = []
    for j in range(len(trophies)):
        y_obs = []
        for _, _, rates in fit_reads:
            if j < len(rates) and rates[j] is not None:
                y_obs.append(logit(rates[j]))
            else:
                y_obs.append(None)
        y_all.append(y_obs)

    # The founders' head start: the first read already shows trophies
    # earned, so the founders were playing before release. Try a few
    # head starts (hours) and keep the one whose fits explain the
    # reads best; 0 means nobody played before release.
    best_a0, best_total = 0.0, None
    for a0 in (0.0, 3.0, 6.0, 12.0):
        ln_age, w_norm = row_matrices(bins_for(a0), fit_t)
        total = 0.0
        for j, tr in enumerate(trophies):
            y_obs = y_all[j]
            if sum(1 for y in y_obs if y is not None) >= 1:
                _, _, raw = fit_trophy(ln_age, w_norm, y_obs,
                                       prior_ln_m(tr), (0.8, 1.2, 1.8), 12)
                total += raw
        if best_total is None or total < best_total:
            best_a0, best_total = a0, total

    ln_age, w_norm = row_matrices(bins_for(best_a0), fit_t)
    trow_a, trow_w = row_matrices(bins_for(best_a0), [t_tgt])
    trow_a, trow_w = trow_a[0], trow_w[0]

    # Last non-null rate per trophy, for the near-certain trophy.
    last_rate = []
    for j in range(len(trophies)):
        rate = None
        for _, _, rates in reversed(reads):
            if j < len(rates) and rates[j] is not None:
                rate = rates[j]
                break
        last_rate.append(rate)

    rows = []
    for j, tr in enumerate(trophies):
        y_obs = y_all[j]
        n_obs = sum(1 for y in y_obs if y is not None)
        ln_m_prior = prior_ln_m(tr)
        if n_obs >= 2:
            observed = [y for y in y_obs if y is not None]
            var = max(observed) - min(observed)
        else:
            var = 0.0
        if n_obs >= 1:
            # Always fit: even a flat history pins the level (the fitted
            # m reproduces the observed rate), and the prior penalty in
            # the search breaks the tie between the (w, m) pairs that
            # match that level. The prior alone is used only when a
            # trophy was never observed at all.
            ln_m, w_use, raw = fit_trophy(ln_age, w_norm, y_obs,
                                          ln_m_prior, W_GRID)
            resid = clamp(math.sqrt(raw / max(n_obs, 1)), 0.05, 0.9)
            if var < 0.05:
                # The level is matched but the shape is unidentified:
                # do not trust the in-sample residual.
                resid = max(resid, 0.35)
        else:
            ln_m = ln_m_prior
            w_use = 1.2
            resid = 0.5
        med_frac = predict_row(trow_a, trow_w, ln_m, w_use)
        med = clamp_rate(med_frac * 100.0)
        # The near-certain trophy (everyone has it): hold it at its rate.
        if last_rate[j] is not None and last_rate[j] >= 99.0:
            med = min(99.99, last_rate[j])
            resid = 0.12
        rows.append((tr["id"], med, resid))

    return build_rows(rows, d_days)


def prior_median_rate(t_target, trophy):
    """Median rate at t_target from the prior alone (no history)."""
    ln_m = prior_ln_m(trophy)
    frac = sigmoid((math.log(max(t_target, 0.05)) - ln_m) / 1.2)
    return clamp_rate(frac * 100.0)


def build_rows(rows, d_days):
    """rows: (id, median %, residual) -> the output object."""
    out = []
    for tid, med, resid in rows:
        lg = logit(med)
        spread = math.sqrt(0.25 * d_days + 0.1225 * d_days +
                           resid * resid + 0.04)
        qs = []
        prev = 0.01
        for z in Z:
            q = clamp_rate(100.0 * sigmoid(lg + z * spread))
            q = round4(max(q, prev))
            qs.append(q)
            prev = q
        out.append({"id": tid, "q": qs})
    return {"trophies": out}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
