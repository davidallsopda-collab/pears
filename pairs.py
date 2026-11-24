"""
Tesla vs SpaceX pair tracker - data for the phone page (docs/pair.html).

Pulls daily prices for TSLA and SPCX since SpaceX listed (12 June 2026),
plus the latest live-ish price, and works out:
  - how closely the two move together and how big SpaceX's moves are vs Tesla's
  - today's gap: did SpaceX move as much as it "should" have given Tesla's move?
  - the stretch score: how far the SPCX/TSLA price ratio is from its usual level
  - history tests of the "when one's up, buy the other" ideas

Used by pears.py (the Pears app). Run on its own it writes docs/pairs.json.
(see .github/workflows/scan.yml). You can also run it on your PC:  python pairs.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(HERE, "docs", "pairs.json")

A, B = "TSLA", "SPCX"          # A = Tesla (the leader we watch), B = SpaceX
NAMES = {A: "Tesla", B: "SpaceX"}
START = "2026-06-12"           # SpaceX's first day of trading
WINDOW = 20                    # trading days used for the "usual" ratio (about a month)
Z_SIGNAL = 2.0                 # stretch score that counts as a signal
BIG_MOVE = 2.0                 # % daily move that counts as "jumped"
LAG_Z = 1.0                    # how unusual today's gap must be to count as lagging


# ---------- when to run ----------

def market_window_now():
    """True on weekdays from 9:25am to 4:30pm New York time."""
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return False
    mins = now.hour * 60 + now.minute
    return 9 * 60 + 25 <= mins <= 16 * 60 + 30


# ---------- data ----------

def load_prices():
    import pandas as pd
    import yfinance as yf
    from zoneinfo import ZoneInfo

    raw = yf.download([A, B], start=START, interval="1d", auto_adjust=True, progress=False)
    df = raw["Close"][[A, B]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()

    # swap in the latest price for today so the page is live during the day
    live = {}
    for sym in (A, B):
        try:
            live[sym] = float(yf.Ticker(sym).fast_info["last_price"])
        except Exception:
            pass
    if len(live) == 2 and market_window_now():
        today = pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date())
        df.loc[today, A] = live[A]
        df.loc[today, B] = live[B]
        df = df.sort_index()
    return df


# ---------- maths ----------

def clean(x, nd=2):
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    return None if math.isnan(x) or math.isinf(x) else round(x, nd)


def verdict(n, win, avg, base):
    if n < 8:
        return "few", "Not enough cases yet to tell"
    if win >= 60 and avg > base:
        return "works", "Has worked so far"
    if win <= 40 and avg < base:
        return "opposite", "Has done the opposite so far"
    return "flip", "No real edge so far - about a coin flip"


def test(name, explain, mask, outcome, baseline, win_word):
    """mask: days the idea fires. outcome: % result you'd have got after each day."""
    hits = outcome[mask].dropna()
    n = int(len(hits))
    win = float((hits > 0).mean() * 100) if n else 0.0
    avg = float(hits.mean()) if n else 0.0
    base = float(baseline.dropna().mean()) if baseline.dropna().size else 0.0
    tone, text = verdict(n, win, avg, base)
    last = [d.strftime("%Y-%m-%d") for d in hits.index[-3:]]
    return {
        "name": name, "explain": explain, "n": n, "win_rate": clean(win, 0),
        "avg": clean(avg), "baseline": clean(base), "win_word": win_word,
        "tone": tone, "verdict": text, "recent": last,
    }


BANDS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]


def rotate(a, b, z, band, from_i=0):
    """Switch strategy: hold SpaceX after the stretch score hits -band, switch to Tesla
    after it hits +band. Decides on each close, trades at that close (no peeking ahead).
    Runs over the whole history so the position carries over; only counts money from
    from_i onwards. Returns growth of 100, closed trades (% each) and what it holds now."""
    hold, cap, start_cap, entry, trades = None, 100.0, None, None, []
    for i in range(len(z) - 1):
        if i == from_i:
            start_cap = cap
            if hold:
                entry = cap               # a position carried in counts from here
        zi = z[i]
        if zi is not None and not math.isnan(zi):
            want = "B" if zi <= -band else "A" if zi >= band else hold
            if want != hold:
                if hold and i >= from_i and entry is not None:
                    trades.append((cap / entry - 1) * 100)
                hold = want
                entry = cap if i >= from_i else None
        if hold:
            p = b if hold == "B" else a
            cap *= p[i + 1] / p[i]
    if start_cap is None:
        start_cap = cap
    if hold and entry is not None:
        trades.append((cap / entry - 1) * 100)     # still open: marked at today's price
    return cap / start_cap * 100, trades, hold


def rotation(df, z):
    a, b = list(df[A]), list(df[B])
    zl = [None if (v is None or math.isnan(v)) else float(v) for v in z]
    first = next((i for i, v in enumerate(zl) if v is not None), None)
    if first is None or len(zl) - first < 30:
        return None
    split = first + (len(zl) - first) // 2
    dates = [d.strftime("%Y-%m-%d") for d in df.index]

    def hold_pct(p, s, e):
        return (p[e] / p[s] - 1) * 100

    # 1) tune on the first half only
    train = []
    for band in BANDS:
        g, t, _ = rotate(a[:split + 1], b[:split + 1], zl[:split + 1], band, from_i=first)
        train.append((g, band, len(t)))
    best_g, best_band, best_n = max(train, key=lambda r: (r[0], -abs(r[1] - 2)))

    # 2) test that setting on the second half it has never seen
    g_test, t_test, _ = rotate(a, b, zl, best_band, from_i=split)
    test_ret = g_test - 100
    last = len(a) - 1
    hold_a = hold_pct(a, split, last)
    hold_b = hold_pct(b, split, last)
    hold_mix = (hold_a + hold_b) / 2

    if len(t_test) < 4:
        tone, text = "few", f"Too few trades to judge ({len(t_test)} in the test year)"
    elif test_ret > max(hold_a, hold_b) and test_ret > 0:
        tone, text = "works", "Held up - beat holding either stock on data it wasn't tuned on"
    elif test_ret < hold_mix - 2:
        tone, text = "opposite", "Fell apart on data it wasn't tuned on"
    elif test_ret > hold_mix + 2:
        best = NAMES[A] if hold_a >= hold_b else NAMES[B]
        tone, text = "flip", f"Mixed - beat holding both, but just holding {best} did better"
    else:
        tone, text = "flip", "No better than just holding both"

    # 3) every setting over the whole history, to show how touchy it is
    table = []
    for band in BANDS:
        g, t, _ = rotate(a, b, zl, band, from_i=first)
        table.append({"band": band, "end": clean(g, 1), "trades": len(t),
                      "wins": sum(1 for x in t if x > 0)})

    _, _, now_hold = rotate(a, b, zl, best_band, from_i=first)
    return {
        "train_from": dates[first], "split": dates[split], "to": dates[last],
        "best_band": best_band, "train_return": clean(best_g - 100, 1), "train_trades": best_n,
        "test_return": clean(test_ret, 1), "test_trades": len(t_test),
        "test_wins": sum(1 for x in t_test if x > 0),
        "hold_a": clean(hold_a, 1), "hold_b": clean(hold_b, 1), "hold_mix": clean(hold_mix, 1),
        "tone": tone, "verdict": text,
        "now_holding": {"A": A, "B": B}.get(now_hold),
        "table": table,
        "hold_all_a": clean(hold_pct(a, first, last), 1), "hold_all_b": clean(hold_pct(b, first, last), 1),
    }


def compute(df):
    import numpy as np

    ra = df[A].pct_change() * 100            # Tesla daily % move
    rb = df[B].pct_change() * 100            # SpaceX daily % move
    both = ra.notna() & rb.notna()

    xa, xb = ra[both], rb[both]

    def beta_of(x, y):
        return float(np.cov(y, x)[0, 1] / np.var(x, ddof=1)) if len(x) > 5 else float("nan")

    beta = beta_of(xa, xb)
    beta30 = beta_of(xa.tail(30), xb.tail(30))
    corr = float(xa.corr(xb))
    corr30 = float(xa.tail(30).corr(xb.tail(30)))

    # today's gap: what SpaceX did vs what Tesla's move says it "should" have done
    resid = rb - beta * ra
    resid_sd = float(resid.std())
    gap_z = resid / resid_sd

    # stretch score: how far the SPCX/TSLA price ratio is from its usual level
    # (the chart shows Tesla / SpaceX, like TradingView's TSLA/SPCX; negative score = SpaceX cheap)
    ratio = df[A] / df[B]
    mid = ratio.rolling(WINDOW).mean()
    sd = ratio.rolling(WINDOW).std()
    z = -(ratio - mid) / sd

    # forward results (from that day's close)
    fwd_b1 = rb.shift(-1)
    fwd_a1 = ra.shift(-1)
    fwd_b3 = (df[B].shift(-3) / df[B] - 1) * 100
    rel5 = ((df[B].shift(-5) / df[B]) - (df[A].shift(-5) / df[A])) * 100   # SpaceX beat Tesla by, over 5 days
    rel1 = fwd_b1 - fwd_a1

    na, nb = NAMES[A], NAMES[B]
    tests = [
        test(f"{na} jumps - buy {nb} next day",
             f"Days {na} closed up {BIG_MOVE:g}% or more. Result = {nb}'s move the next day.",
             ra >= BIG_MOVE, fwd_b1, fwd_b1, f"{nb} up next day"),
        test(f"{na} jumps - hold {nb} 3 days",
             f"Same days, but holding {nb} for 3 days instead of 1.",
             ra >= BIG_MOVE, fwd_b3, fwd_b3, f"{nb} up after 3 days"),
        test(f"{nb} jumps - buy {na} next day",
             f"Days {nb} closed up {BIG_MOVE:g}% or more. Result = {na}'s move the next day.",
             rb >= BIG_MOVE, fwd_a1, fwd_a1, f"{na} up next day"),
        test(f"{nb} lagged {na} - does it catch up?",
             f"Days {nb} did noticeably worse than {na}'s move said it should. "
             f"Result = how much {nb} beat {na} the next day.",
             gap_z <= -LAG_Z, rel1, rel1, f"{nb} beat {na} next day"),
        test(f"Stretch score very low - {nb} cheap vs {na}",
             f"Days the stretch score was -{Z_SIGNAL:g} or lower. "
             f"Result = how much {nb} beat {na} over the next 5 days.",
             z <= -Z_SIGNAL, rel5, rel5, f"{nb} beat {na} over 5 days"),
        test(f"Stretch score very high - {nb} dear vs {na}",
             f"Days the stretch score was +{Z_SIGNAL:g} or higher. "
             f"Result = how much {na} beat {nb} over the next 5 days.",
             z >= Z_SIGNAL, -rel5, -rel5, f"{na} beat {nb} over 5 days"),
    ]

    zt = clean(z.iloc[-1])
    if zt is None:
        signal = ("none", "Not enough history yet")
    elif zt <= -Z_SIGNAL:
        signal = ("cheap", f"{nb} looks cheap vs {na}")
    elif zt >= Z_SIGNAL:
        signal = ("dear", f"{nb} looks expensive vs {na}")
    elif abs(zt) >= 1:
        signal = ("lean", "Drifting from normal - worth watching")
    else:
        signal = ("normal", "Normal - nothing stretched")

    gt = clean(gap_z.iloc[-1])
    if gt is None:
        today_tag = "none"
    elif gt <= -LAG_Z:
        today_tag = "lagging"
    elif gt >= LAG_Z:
        today_tag = "ahead"
    else:
        today_tag = "inline"

    last_date = df.index[-1]
    return {
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "a": A, "b": B, "names": NAMES,
        "window": WINDOW, "z_signal": Z_SIGNAL, "big_move": BIG_MOVE,
        "days": int(both.sum()),
        "last_date": last_date.strftime("%Y-%m-%d"),
        "price": {A: clean(df[A].iloc[-1]), B: clean(df[B].iloc[-1])},
        "today": {
            "a_move": clean(ra.iloc[-1]), "b_move": clean(rb.iloc[-1]),
            "b_expected": clean(beta * ra.iloc[-1]), "gap": clean(resid.iloc[-1]),
            "gap_z": gt, "tag": today_tag,
        },
        "stats": {"corr": clean(corr), "corr30": clean(corr30), "beta": clean(beta), "beta30": clean(beta30)},
        "z": zt, "signal": {"code": signal[0], "text": signal[1]},
        "tests": tests,
        "rotation": rotation(df, z),
        "history": {
            "dates": [d.strftime("%Y-%m-%d") for d in df.index],
            A: [clean(v) for v in df[A]],
            B: [clean(v) for v in df[B]],
            "z": [clean(v) for v in z],
            "ratio": [clean(v, 4) for v in ratio],
            "mid": [clean(v, 4) for v in mid],
            "sd": [clean(v, 5) for v in sd],
        },
    }


def main():
    is_scheduled = os.environ.get("GITHUB_EVENT_NAME") == "schedule"
    if is_scheduled and not market_window_now():
        print("Market closed - skipping pair update.")
        return 0
    df = load_prices()
    if len(df) < WINDOW + 5:
        print(f"Only {len(df)} days of prices came back - keeping the old file.")
        return 1
    out = compute(df)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"Saved {OUT_FILE}: {out['days']} days, stretch {out['z']}, {out['signal']['text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
