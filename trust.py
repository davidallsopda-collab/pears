"""
Trust test for every pair - a tougher, more honest check than the app's one split.

For each pair (up to 5 years of daily prices):
  1. Walk forward: every 3 months, pick the trigger level using ONLY the past,
     then trade the next 3 months blind. The position carries over between quarters.
  2. Plateau, not peak: the chosen level is the one whose neighbours also did well.
  3. Costs: every switch is charged COST (buy/sell gap + currency fees, a rough guess).
  4. Monkey test: 1,000 random switchers with the same number of switches, on random
     days, same costs. Trust = % of monkeys the rule beat.
  5. Year by year: in how many calendar years did it beat holding both 50/50?

Writes docs/trust.json for the app (trust badges). GitHub runs it once a week.
Run on your PC:  python trust.py
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs")
YEARS = 5
WINDOW = 20
BANDS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
COST = 0.002          # 0.2% per switch
WARMUP = 252          # first year only used for tuning
STEP = 63             # re-tune every 3 months
MONKEYS = 1000
rng = np.random.default_rng(42)


def sim(a, b, z, band, start, end, hold=None, cost=COST):
    """Switch rule from day start to day end. Returns growth (x), final hold, switch count."""
    cap, n = 1.0, 0
    for i in range(start, end):
        zi = z[i]
        if not math.isnan(zi):
            want = "B" if zi <= -band else "A" if zi >= band else hold
            if want != hold:
                if want is not None:
                    cap *= (1 - cost)
                    n += 1
                hold = want
        if hold:
            p = b if hold == "B" else a
            cap *= p[i + 1] / p[i]
    return cap, hold, n


def pick_band(a, b, z, first, upto):
    """Score each level on the past, then smooth with its neighbours (plateau)."""
    g = [sim(a, b, z, x, first, upto)[0] for x in BANDS]
    plateau = []
    for i in range(len(BANDS)):
        nb = g[max(0, i - 1): i + 2]
        plateau.append(sum(nb) / len(nb))
    return BANDS[int(np.argmax(plateau))], g


def walk_forward(a, b, z, first):
    n = len(a)
    start = first + WARMUP
    if start >= n - STEP:
        return None
    # state going into the test period (what the rule would already be holding)
    band0, _ = pick_band(a, b, z, first, start)
    _, hold, _ = sim(a, b, z, band0, first, start)
    cap, switches, bands_used, curve, switch_days = 1.0, 0, [], [1.0], []
    i = start
    while i < n - 1:
        band, _ = pick_band(a, b, z, first, i)
        bands_used.append(band)
        end = min(i + STEP, n - 1)
        for d in range(i, end):
            zi = z[d]
            if not math.isnan(zi):
                want = "B" if zi <= -band else "A" if zi >= band else hold
                if want != hold and want is not None:
                    cap *= (1 - COST)
                    switches += 1
                    switch_days.append(d)
                    hold = want
            if hold:
                p = b if hold == "B" else a
                cap *= p[d + 1] / p[d]
            curve.append(cap)
        i = end
    return {"start": start, "growth": cap, "switches": switches, "bands": bands_used, "curve": curve}


def monkeys(a, b, start, k):
    """Random switchers: same number of switches (k), random days, random first holding."""
    n = len(a)
    la = np.diff(np.log(a[start:]))
    lb = np.diff(np.log(b[start:]))
    CA = np.concatenate([[0], np.cumsum(la)])
    CB = np.concatenate([[0], np.cumsum(lb)])
    days = len(la)
    out = np.empty(MONKEYS)
    k = max(1, k)
    for m in range(MONKEYS):
        cuts = np.sort(rng.choice(np.arange(1, days), size=min(k, days - 1), replace=False))
        edges = np.concatenate([[0], cuts, [days]])
        on_a = rng.random() < 0.5
        tot = 0.0
        for s, e in zip(edges[:-1], edges[1:]):
            tot += (CA[e] - CA[s]) if on_a else (CB[e] - CB[s])
            on_a = not on_a
        out[m] = math.exp(tot) * (1 - COST) ** (len(cuts) + 1)
    return out


def yearly(dates, curve, a, b, start):
    """Per calendar year: rule % vs 50/50 hold-both %."""
    years = {}
    for j, d in enumerate(dates[start:]):
        years.setdefault(d[:4], []).append(j)
    res = []
    for y, idx in years.items():
        s, e = idx[0], idx[-1]
        if e - s < 40:
            continue
        rule = (curve[e] / curve[s] - 1) * 100
        mix = ((a[start + e] / a[start + s] + b[start + e] / b[start + s]) / 2 - 1) * 100
        res.append({"year": y, "rule": round(float(rule), 1), "both": round(float(mix), 1), "beat": bool(rule > mix)})
    return res


def badge(r):
    """green = trusted, amber = maybe, red = fluke / no edge."""
    edge = r["rule_pct"] - r["both_pct"]
    yrs_ok = r["years_total"] and r["years_beat_both"] / r["years_total"] >= 0.75
    if r["beats_monkeys"] >= 90 and edge >= 10 and yrs_ok:
        return "green"
    if r["beats_monkeys"] >= 70 and edge >= 3:
        return "amber"
    return "red"


def label(p):
    if p >= 90:
        return "strong"
    if p >= 75:
        return "promising"
    if p >= 50:
        return "meh"
    return "no edge"


def main():
    import pandas as pd
    import yfinance as yf

    plist = json.load(open(os.path.join(HERE, "pears_list.json"), encoding="utf-8"))
    syms = sorted({s for p in plist for s in (p["a"], p["b"])})
    start = (datetime.now() - timedelta(days=int(365 * YEARS) + 60)).strftime("%Y-%m-%d")
    close = yf.download(syms, start=start, interval="1d", auto_adjust=True, progress=False)["Close"]
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()

    rows = []
    for p in plist:
        A, B = p["a"], p["b"]
        df = close[[A, B]].dropna()
        df = df[df.index >= df.index[-1] - pd.Timedelta(days=int(365 * YEARS))]
        a, b = df[A].to_numpy(float), df[B].to_numpy(float)
        dates = [d.strftime("%Y-%m-%d") for d in df.index]
        ratio = pd.Series(a) / pd.Series(b)
        z = (-(ratio - ratio.rolling(WINDOW).mean()) / ratio.rolling(WINDOW).std()).to_numpy(float)
        first = int(np.argmax(~np.isnan(z)))
        wf = walk_forward(a, b, z, first)
        if not wf:
            print(f"{A}/{B}: not enough history")
            rows.append({"id": f"{A}-{B}", "names": [p["na"], p["nb"]], "kind": p.get("kind"), "skip": True,
                         "badge": "new", "years": round(len(a) / 252, 1)})
            continue
        s = wf["start"]
        g = wf["growth"]
        hold_a, hold_b = a[-1] / a[s], b[-1] / b[s]
        mk = monkeys(a, b, s, wf["switches"])
        pct = float((g > mk).mean() * 100)
        yrs = yearly(dates, wf["curve"], a, b, s)
        test_years = (len(a) - 1 - s) / 252
        cagr = (g ** (1 / test_years) - 1) * 100 if test_years > 0 else 0
        row = {
            "id": f"{A}-{B}", "names": [p["na"], p["nb"]], "kind": p.get("kind"),
            "from": dates[s], "to": dates[-1], "test_years": round(test_years, 1),
            "rule_pct": round((g - 1) * 100, 1), "rule_per_year": round(cagr, 1),
            "hold_a_pct": round((hold_a - 1) * 100, 1), "hold_b_pct": round((hold_b - 1) * 100, 1),
            "both_pct": round(((hold_a + hold_b) / 2 - 1) * 100, 1),
            "switches": wf["switches"], "switches_per_year": round(wf["switches"] / test_years, 1),
            "monkey_median_pct": round((float(np.median(mk)) - 1) * 100, 1),
            "beats_monkeys": round(pct), "trust": label(pct),
            "years_beat_both": sum(1 for y in yrs if y["beat"]), "years_total": len(yrs), "by_year": yrs,
            "bands_used": sorted(set(wf["bands"])),
            "band_now": wf["bands"][-1],
        }
        row["badge"] = badge(row)
        rows.append(row)
        print(f"{row['badge']:<6} {row['id']:<11} {row['test_years']}y  rule {row['rule_pct']:>7}%  both {row['both_pct']:>7}%  "
              f"monkeys beaten {row['beats_monkeys']:>3}%  years {row['years_beat_both']}/{row['years_total']}  "
              f"switches/yr {row['switches_per_year']}")

    os.makedirs(OUT, exist_ok=True)
    meta = {"updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "cost_per_switch": COST,
            "window": WINDOW, "warmup_days": WARMUP, "retune_days": STEP, "monkeys": MONKEYS}
    json.dump({"meta": meta, "pairs": rows}, open(os.path.join(OUT, "trust.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
