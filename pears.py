"""
Pears - pair trade data for the Pears app (docs/).

Reads the pair list from pears_list.json, pulls up to 2 years of daily prices
for every stock in one go, and runs the same maths as the Musk Pair page
(pairs.py) on each pair. Saves:
  docs/data/<A>-<B>.json   full detail for one pair (chart, tests, today)
  docs/summary.json        one line per pair for the home screen

GitHub runs it every 15 minutes while the US market is open.
You can also run it on your PC:  python pears.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone, timedelta

import pairs   # the maths lives there

HERE = os.path.dirname(os.path.abspath(__file__))
LIST_FILE = os.path.join(HERE, "pears_list.json")
OUT_DIR = os.path.join(HERE, "docs")
YEARS = 2


def half_life(ratio, mid):
    """Roughly how many trading days a stretch takes to halve (smaller = snaps back faster)."""
    dev = (ratio - mid).dropna()
    if len(dev) < 30:
        return None
    x, y = dev.iloc[:-1].values, dev.iloc[1:].values
    phi = float((x * y).sum() / (x * x).sum())
    if not 0 < phi < 1:
        return None
    return round(-math.log(2) / math.log(phi), 1)


def load_all(symbols):
    import pandas as pd
    import yfinance as yf
    from zoneinfo import ZoneInfo

    start = (datetime.now() - timedelta(days=int(365 * YEARS) + 40)).strftime("%Y-%m-%d")
    raw = yf.download(sorted(set(symbols)), start=start, interval="1d", auto_adjust=True, progress=False)
    close = raw["Close"]
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()

    if pairs.market_window_now():
        today = pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date())
        for sym in set(symbols):
            try:
                close.loc[today, sym] = float(yf.Ticker(sym).fast_info["last_price"])
            except Exception:
                pass
        close = close.sort_index()
    return close


def build(close, p):
    import pandas as pd
    a, b = p["a"], p["b"]
    df = close[[a, b]].dropna()
    cutoff = df.index[-1] - pd.Timedelta(days=int(365 * YEARS))
    df = df[df.index >= cutoff]
    if len(df) < pairs.WINDOW + 30:
        raise ValueError(f"only {len(df)} days of prices")

    pairs.A, pairs.B = a, b
    pairs.NAMES = {a: p["na"], b: p["nb"]}
    out = pairs.compute(df)

    ratio = df[a] / df[b]
    mid = ratio.rolling(pairs.WINDOW).mean()
    out["id"] = f"{a}-{b}"
    out["group"] = p.get("group", "")
    out["kind"] = p.get("kind", "twins")
    out["half_life"] = half_life(ratio, mid)
    return out


def summary_row(d):
    r = d.get("rotation") or {}
    h = d["history"]
    # 60-day mini line of the ratio for the home screen
    spark = [v for v in h["ratio"][-60:]]
    return {
        "id": d["id"], "a": d["a"], "b": d["b"], "names": d["names"], "group": d["group"], "kind": d.get("kind", "twins"),
        "days": d["days"], "from": h["dates"][0],
        "z": d["z"], "signal": d["signal"],
        "corr": d["stats"]["corr"], "corr30": d["stats"]["corr30"], "half_life": d["half_life"],
        "best_band": r.get("best_band"), "now_holding": r.get("now_holding"),
        "test_return": r.get("test_return"), "hold_a": r.get("hold_a"), "hold_b": r.get("hold_b"),
        "hold_mix": r.get("hold_mix"), "tone": r.get("tone"), "verdict": r.get("verdict"),
        "test_trades": r.get("test_trades"), "test_wins": r.get("test_wins"),
        "today": d["today"], "spark": spark,
    }


def main():
    is_scheduled = os.environ.get("GITHUB_EVENT_NAME") == "schedule"
    if is_scheduled and not pairs.market_window_now():
        print("Market closed - skipping Pears update.")
        return 0

    with open(LIST_FILE, encoding="utf-8") as f:
        plist = json.load(f)
    close = load_all([s for p in plist for s in (p["a"], p["b"])])

    os.makedirs(os.path.join(OUT_DIR, "data"), exist_ok=True)
    rows = []
    for p in plist:
        try:
            d = build(close, p)
        except Exception as e:
            print(f"{p['a']}/{p['b']} failed: {e}")
            continue
        with open(os.path.join(OUT_DIR, "data", d["id"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(d, f, separators=(",", ":"))
        rows.append(summary_row(d))
        print(f"{d['id']}: {d['days']} days, in step {d['stats']['corr']}, test {rows[-1]['test_return']} ({rows[-1]['verdict']})")

    if not rows:
        print("No pairs worked - keeping the old files.")
        return 1
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "pairs": rows},
                  f, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
