"""Track the model's pre-fight picks against real results.

Two commands:

  log    Record predictions for an upcoming card BEFORE it happens.
         python scripts/track_predictions.py log --event "UFC 333: X vs. Y" \
             --date 2026-10-17 --fights "Fighter A|Fighter B;Fighter C|Fighter D" \
             [--model-dir .] [--model-note "..."]
         Names must be the database spellings (fighters_clean.csv). Fights
         already logged for that event are skipped, so re-running is safe.

  score  Fill in results for logged fights from the Greco1899 dataset and
         rewrite tracking/scorecard.md with the running tally.
         python scripts/track_predictions.py score [--data-dir data/scrape_ufc_stats]

Files (in tracking/):
  predictions_log.csv  one row per predicted fight, with result once known
  scorecard.md         running tally: overall, by confidence, by event
"""
import argparse
import os
import sys
from datetime import datetime

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
TRACK = os.path.join(PROJECT, "tracking")
LOG = os.path.join(TRACK, "predictions_log.csv")
CARD = os.path.join(TRACK, "scorecard.md")
COLS = ["event", "event_date", "fighter_a", "fighter_b", "pick", "confidence", "model",
        "logged_at", "winner", "method", "correct"]


def load_log():
    if os.path.exists(LOG):
        df = pd.read_csv(LOG, dtype={"correct": str, "winner": str, "method": str})
        df[["winner", "method", "correct"]] = df[["winner", "method", "correct"]].fillna("")
        return df
    return pd.DataFrame(columns=COLS)


def cmd_log(a):
    os.makedirs(TRACK, exist_ok=True)
    model_dir = os.path.abspath(a.model_dir)
    sys.path.insert(0, PROJECT)
    os.chdir(model_dir)  # ufc_predictor loads model files from the working directory
    import warnings
    warnings.filterwarnings("ignore")
    import ufc_predictor as up

    up._load_resources()
    model = a.model_note or f"{up._model_name} ({datetime.fromtimestamp(os.path.getmtime(up._model_name)):%Y-%m-%d})"
    log = load_log()
    done = {(r.fighter_a, r.fighter_b) for r in log[log.event == a.event].itertuples()}
    rows, skipped = [], []
    for pair in [p for p in a.fights.split(";") if p.strip()]:
        fa, fb = [x.strip() for x in pair.split("|")]
        if (fa, fb) in done or (fb, fa) in done:
            continue
        p = up.predict_fight(fa, fb, verbose=False)
        if p is None or p != p:
            skipped.append(f"{fa} vs {fb}")
            continue
        rows.append(dict(event=a.event, event_date=a.date, fighter_a=fa, fighter_b=fb,
                         pick=fa if p > 0.5 else fb, confidence=round(max(p, 1 - p), 3), model=model,
                         logged_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
                         winner="", method="", correct=""))
    if rows:
        log = pd.concat([log, pd.DataFrame(rows)], ignore_index=True)
        log.to_csv(LOG, index=False)
    print(f"Logged {len(rows)} predictions for {a.event} using {model}")
    for r in rows:
        print(f"  {r['fighter_a']} vs {r['fighter_b']}: {r['pick']} {r['confidence']:.0%}")
    if skipped:
        print("Could not predict (name not in DB?): " + "; ".join(skipped))
    return 0


def cmd_score(a):
    log = load_log()
    if log.empty:
        print("No predictions logged yet.")
        return 0
    res = pd.read_csv(os.path.join(a.data_dir, "ufc_fight_results.csv"))
    ev = pd.read_csv(os.path.join(a.data_dir, "ufc_event_details.csv"))
    for df in (res, ev):
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].str.strip()
    res["date"] = res["EVENT"].map(ev.set_index("EVENT")["DATE"].pipe(pd.to_datetime, errors="coerce"))
    names = res["BOUT"].str.split(r"\s+vs\.\s+", n=1, expand=True)
    res["A"], res["B"] = names[0].str.strip(), names[1].str.strip()

    newly = 0
    for i, r in log.iterrows():
        if isinstance(r["winner"], str) and r["winner"]:
            continue
        d = pd.to_datetime(r["event_date"])
        pair = {r["fighter_a"], r["fighter_b"]}
        m = res[(res["date"].sub(d).abs() <= pd.Timedelta(days=3))
                & res.apply(lambda x: {x["A"], x["B"]} == pair, axis=1)]
        if m.empty:
            continue
        x = m.iloc[0]
        o = x["OUTCOME"]
        if o == "W/L":
            w = x["A"]
        elif o == "L/W":
            w = x["B"]
        else:
            w = "Draw" if o == "D/D" else "No contest"
        log.at[i, "winner"] = w
        log.at[i, "method"] = str(x["METHOD"]).strip()
        log.at[i, "correct"] = "" if w in ("Draw", "No contest") else str(w == r["pick"])
        newly += 1
    log.to_csv(LOG, index=False)
    write_scorecard(log)
    print(f"Scored {newly} newly finished fights. Scorecard: {CARD}")
    print(open(CARD).read())
    return 0


def write_scorecard(log):
    s = log[log["correct"].isin(["True", "False"])].copy()
    s["hit"] = s["correct"] == "True"
    pending = log[(log["winner"].isna()) | (log["winner"] == "")]
    lines = ["# UFC model — running scorecard", "",
             f"_Updated {datetime.now():%Y-%m-%d %H:%M}. Only predictions logged before each event count._", ""]
    if s.empty:
        lines.append("No finished fights yet.")
    else:
        lines += [f"**Overall: {s.hit.sum()} of {len(s)} correct ({s.hit.mean():.1%})**", "",
                  "| Confidence | Fights | Correct | Hit rate |", "|---|---|---|---|"]
        for lo, hi, lab in [(0.5, 0.6, "50–59%"), (0.6, 0.65, "60–64%"), (0.65, 0.75, "65–74%"), (0.75, 1.01, "75%+")]:
            b = s[(s.confidence >= lo) & (s.confidence < hi)]
            if len(b):
                lines.append(f"| {lab} | {len(b)} | {b.hit.sum()} | {b.hit.mean():.0%} |")
        lines += ["", "| Event | Date | Correct | Hit rate | Model |", "|---|---|---|---|---|"]
        for (e, d), g in s.groupby(["event", "event_date"], sort=False):
            lines.append(f"| {e} | {d} | {g.hit.sum()} of {len(g)} | {g.hit.mean():.0%} | {g.model.iloc[0]} |")
        draws = log[log["winner"].isin(["Draw", "No contest"])]
        if len(draws):
            lines += ["", f"Draws/no contests (not counted): {len(draws)}"]
    if len(pending):
        lines += ["", f"Awaiting results: {len(pending)} fights "
                      f"({', '.join(sorted(pending['event'].unique()))})"]
    with open(CARD, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("log")
    l.add_argument("--event", required=True)
    l.add_argument("--date", required=True)
    l.add_argument("--fights", required=True)
    l.add_argument("--model-dir", default=PROJECT)
    l.add_argument("--model-note", default=None)
    sc = sub.add_parser("score")
    sc.add_argument("--data-dir", default=os.path.join(PROJECT, "data", "scrape_ufc_stats"))
    a = ap.parse_args()
    return cmd_log(a) if a.cmd == "log" else cmd_score(a)


if __name__ == "__main__":
    sys.exit(main())
