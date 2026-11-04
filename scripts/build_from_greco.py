"""Build fighters_clean.csv and matches_clean.csv from the Greco1899 dataset.

Replaces UFC_scraping.ipynb + scrape_ufc_matches.py + UFC_Cleaning.ipynb, which
can no longer fetch ufcstats.com directly (the site now serves a browser check
to scripts). Source: https://github.com/Greco1899/scrape_ufc_stats — a public
dataset of ufcstats.com data refreshed almost daily.

Output columns match what UFC_Machine_Learning_Model.ipynb and ufc_predictor.py
expect, so neither needs changes.

Differences from the old pipeline:
  * Career stats (SLpM, Str. Acc., ...) are computed from round-by-round fight
    stats using ufcstats.com's definitions, rather than read off fighter pages.
  * Wins/Losses/Draws count UFC fights only (the old data used full pro records).
  * Fight order in matches_clean.csv is randomized per fight (seeded, so it's
    reproducible) so the training data isn't biased toward the listed winner.
  * Draws and no-contests are excluded from matches_clean.csv.

Usage:
    python scripts/build_from_greco.py [--data-dir data/scrape_ufc_stats]
                                       [--max-age-days 45]

Exit codes: 0 ok, 1 error, 2 dataset is stale (latest event too old).
"""
import argparse
import hashlib
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.path.join(PROJECT, "data", "scrape_ufc_stats"))
    p.add_argument("--out-dir", default=PROJECT)
    p.add_argument("--max-age-days", type=int, default=45,
                   help="Fail with exit code 2 if the newest event is older than this")
    p.add_argument("--today", default=None, help="Override 'today' (YYYY-MM-DD), for testing")
    return p.parse_args()


def strip_cols(df):
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip()
    return df


def landed_attempted(series):
    """'29 of 73' -> (29, 73). Missing/odd values -> (0, 0)."""
    parts = series.fillna("0 of 0").astype(str).str.extract(r"(\d+)\s*of\s*(\d+)")
    return parts[0].fillna(0).astype(int), parts[1].fillna(0).astype(int)


def fight_minutes(row):
    """Total fight time in minutes from ROUND, TIME and TIME FORMAT."""
    try:
        rnd = int(row["ROUND"])
        m, s = str(row["TIME"]).split(":")
        last = int(m) + int(s) / 60
    except (ValueError, TypeError):
        return np.nan
    fmt = str(row["TIME FORMAT"])
    lengths = re.search(r"\(([\d\-]+)\)", fmt)
    if lengths:
        rl = [int(x) for x in lengths.group(1).split("-")]
        prior = sum(rl[: rnd - 1]) if rnd - 1 <= len(rl) else sum(rl) + 5 * (rnd - 1 - len(rl))
    else:  # e.g. "No Time Limit" — assume nothing before the final round
        prior = 0
    return prior + last


def stable_coin(key):
    """Deterministic 0/1 from a string, so corner order is reproducible."""
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 2


STAT_COLS = ["minutes", "sig_l", "sig_a", "o_sig_l", "o_sig_a",
             "td_l", "td_a", "o_td_l", "o_td_a", "sub"]

# Feature order the model is trained on — must match ufc_predictor.py
FEATURES = ["Weight:", "Age", "SLpM:", "Str. Acc.:", "SApM:", "Str. Def:",
            "TD Avg.:", "TD Acc.:", "TD Def.:", "Sub. Avg.:", "Reach:",
            "Win Percentage", "Total Fights", "Height:_inches"]


def _ratio(n, d_):
    n, d_ = np.asarray(n, dtype=float), np.asarray(d_, dtype=float)
    return np.divide(n, d_, out=np.zeros_like(n), where=d_ > 0)


def career_stats(tot):
    """ufcstats.com career-stat definitions from summed fight totals."""
    c = pd.DataFrame(index=tot.index)
    c["SLpM:"] = _ratio(tot.sig_l, tot.minutes)
    c["Str. Acc.:"] = _ratio(tot.sig_l, tot.sig_a)
    c["SApM:"] = _ratio(tot.o_sig_l, tot.minutes)
    c["Str. Def:"] = np.where(tot.o_sig_a > 0, 1 - _ratio(tot.o_sig_l, tot.o_sig_a), 0.0)
    c["TD Avg.:"] = _ratio(tot.td_l, tot.minutes) * 15
    c["TD Acc.:"] = _ratio(tot.td_l, tot.td_a)
    c["TD Def.:"] = np.where(tot.o_td_a > 0, 1 - _ratio(tot.o_td_l, tot.o_td_a), 0.0)
    c["Sub. Avg.:"] = _ratio(tot["sub"], tot.minutes) * 15
    return c.round(2)


def height_inches(h):
    m = re.match(r"(\d+)'\s*(\d+)", str(h))
    return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan


def build_point_in_time(res, per, ev, f, dobs):
    """One training row per decided fight, using each fighter's stats and record
    from *earlier* UFC events only — what was knowable before the fight.

    res: fight results with A, B, OUTCOME, EVENT, BOUT, minutes, URL, Red, Blue, Winner
    per: per-fighter per-fight stat totals (EVENT, BOUT, FIGHTER, STAT_COLS...)
    f:   fighters frame indexed by Name (static Weight:, Reach:, Height:)
    dobs: Series Name -> date of birth
    """
    dates = ev.set_index("EVENT")["DATE"]
    res = res.copy()
    res["date"] = res["EVENT"].map(dates)
    res = res.dropna(subset=["date", "A", "B"])

    long = pd.concat([
        pd.DataFrame({"FIGHTER": res["A"], "EVENT": res["EVENT"], "BOUT": res["BOUT"], "date": res["date"],
                      "win": (res["OUTCOME"] == "W/L").astype(int), "loss": (res["OUTCOME"] == "L/W").astype(int),
                      "draw": (res["OUTCOME"] == "D/D").astype(int)}),
        pd.DataFrame({"FIGHTER": res["B"], "EVENT": res["EVENT"], "BOUT": res["BOUT"], "date": res["date"],
                      "win": (res["OUTCOME"] == "L/W").astype(int), "loss": (res["OUTCOME"] == "W/L").astype(int),
                      "draw": (res["OUTCOME"] == "D/D").astype(int)}),
    ], ignore_index=True)
    long = long.merge(per[["EVENT", "BOUT", "FIGHTER"] + STAT_COLS], on=["EVENT", "BOUT", "FIGHTER"], how="left")
    long[STAT_COLS] = long[STAT_COLS].fillna(0)

    cols = STAT_COLS + ["win", "loss", "draw"]
    day = long.groupby(["FIGHTER", "date"])[cols].sum().sort_index()
    prior = day.groupby(level=0).cumsum() - day  # totals from strictly earlier dates

    feats = career_stats(prior)
    tf_ = prior["win"] + prior["loss"] + prior["draw"]
    feats["Total Fights"] = tf_
    feats["Win Percentage"] = _ratio(prior["win"], tf_).round(2)

    wl = res[res["OUTCOME"].isin(["W/L", "L/W"])]
    out = pd.DataFrame({"date": wl["date"].values, "Red": wl["Red"].values, "Blue": wl["Blue"].values,
                        "result": (wl["Winner"] == wl["Red"]).astype(int).values})
    h = f["Height:"].map(height_inches)
    for side in ("Red", "Blue"):
        key = pd.MultiIndex.from_arrays([out[side], out["date"]])
        fs = feats.reindex(key)
        fs.index = out.index
        fs["Weight:"] = out[side].map(f["Weight:"]).values
        fs["Reach:"] = out[side].map(f["Reach:"]).values
        fs["Height:_inches"] = out[side].map(h).values
        fs["Age"] = ((out["date"] - out[side].map(dobs)).dt.days / 365.25).round(1).values
        out[[f"{side}_{c}" for c in FEATURES]] = fs[FEATURES].values
    for c in FEATURES:
        out[c] = out[f"Red_{c}"] - out[f"Blue_{c}"]
    return out[["date", "Red", "Blue", "result"] + FEATURES].sort_values("date").reset_index(drop=True)


SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def norm_tokens(name):
    import unicodedata
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[\-\.'’,]", " ", s)
    return [t for t in s.split() if t and t not in SUFFIXES]


def build_name_map(fight_names, profile_names):
    """Map fighter-profile names to the names used on fight cards.

    ufcstats spells some fighters differently on their profile page vs. their
    fights (e.g. 'Waldo Cortes-Acosta' vs 'Waldo Cortes Acosta', 'Michael Aswell'
    vs 'Michael Aswell Jr.'). Fight-card spelling wins, since that's what you type
    when predicting a card.
    """
    profiles = set(profile_names)
    by_key, by_sorted = {}, {}
    for p in profiles:
        t = norm_tokens(p)
        by_key.setdefault(" ".join(t), []).append(p)
        by_sorted.setdefault(" ".join(sorted(t)), []).append(p)
    mapping = {}
    used = set(fight_names) & profiles
    for fn in sorted(set(fight_names) - profiles):
        t = norm_tokens(fn)
        cands = by_key.get(" ".join(t)) or by_sorted.get(" ".join(sorted(t))) or []
        if not cands and len(t) >= 2:  # e.g. 'Ilimbek Akylbek' vs 'Ilimbek Akylbek Uulu'
            ts = set(t)
            cands = [p for p in profiles
                     if len(ts & set(norm_tokens(p))) >= 2
                     and (ts <= set(norm_tokens(p)) or set(norm_tokens(p)) <= ts)]
        cands = [c for c in cands if c not in used]
        if len(cands) == 1:
            mapping[cands[0]] = fn
            used.add(cands[0])
    return mapping


def clean_weightclass(wc):
    wc = str(wc).replace("UFC ", "").replace(" Title", "").replace(" Bout", "")
    wc = re.sub(r"\s+", " ", wc).strip()
    return wc or "Unknown"


def main():
    a = parse_args()
    today = pd.Timestamp(a.today) if a.today else pd.Timestamp(datetime.today().date())
    d = a.data_dir
    need = ["ufc_fighter_tott.csv", "ufc_fighter_details.csv", "ufc_fight_results.csv",
            "ufc_fight_stats.csv", "ufc_event_details.csv"]
    missing = [f for f in need if not os.path.exists(os.path.join(d, f))]
    if missing:
        print(f"ERROR: missing files in {d}: {missing}")
        return 1

    tott = strip_cols(pd.read_csv(os.path.join(d, "ufc_fighter_tott.csv")))
    det = strip_cols(pd.read_csv(os.path.join(d, "ufc_fighter_details.csv")))
    res = strip_cols(pd.read_csv(os.path.join(d, "ufc_fight_results.csv")))
    st = strip_cols(pd.read_csv(os.path.join(d, "ufc_fight_stats.csv")))
    ev = strip_cols(pd.read_csv(os.path.join(d, "ufc_event_details.csv")))

    # ---------- Freshness check ----------
    ev["DATE"] = pd.to_datetime(ev["DATE"], errors="coerce")
    latest = ev["DATE"].max()
    age_days = (today - latest).days
    latest_name = ev.loc[ev["DATE"].idxmax(), "EVENT"]
    print(f"Dataset: {len(tott)} fighters, {len(res)} fights, {len(ev)} events")
    print(f"Newest event: {latest_name} on {latest.date()} ({age_days} days ago)")
    if age_days > a.max_age_days:
        print(f"STALE: newest event is older than {a.max_age_days} days — "
              "the source dataset may have stopped updating.")
        return 2

    # ---------- Fight durations ----------
    res["minutes"] = res.apply(fight_minutes, axis=1)
    dur = res.set_index(["EVENT", "BOUT"])["minutes"]
    dur = dur[~dur.index.duplicated()]

    # ---------- Per-fighter, per-fight stat totals ----------
    st = st.dropna(subset=["FIGHTER"])
    st = st.drop_duplicates(["EVENT", "BOUT", "ROUND", "FIGHTER"])
    st["sig_l"], st["sig_a"] = landed_attempted(st["SIG.STR."])
    st["td_l"], st["td_a"] = landed_attempted(st["TD"])
    st["sub"] = pd.to_numeric(st["SUB.ATT"], errors="coerce").fillna(0)
    per = st.groupby(["EVENT", "BOUT", "FIGHTER"], as_index=False)[
        ["sig_l", "sig_a", "td_l", "td_a", "sub"]].sum()

    # Attach the opponent's numbers (the other fighter in the same bout)
    opp = per.rename(columns={"FIGHTER": "OPP", "sig_l": "o_sig_l", "sig_a": "o_sig_a",
                              "td_l": "o_td_l", "td_a": "o_td_a", "sub": "o_sub"})
    per = per.merge(opp, on=["EVENT", "BOUT"])
    per = per[per["FIGHTER"] != per["OPP"]]
    per["minutes"] = [dur.get((e, b), np.nan) for e, b in zip(per["EVENT"], per["BOUT"])]
    per = per.dropna(subset=["minutes"])
    per = per[per["minutes"] > 0]

    tot = per.groupby("FIGHTER")[["minutes", "sig_l", "sig_a", "o_sig_l", "o_sig_a",
                                  "td_l", "td_a", "o_td_l", "o_td_a", "sub"]].sum()

    career = career_stats(tot)

    # ---------- UFC records ----------
    names = res["BOUT"].str.split(r"\s+vs\.\s+", n=1, expand=True)
    res["A"], res["B"] = names[0].str.strip(), names[1].str.strip()
    rec = {}

    def bump(n, k):
        rec.setdefault(n, {"Wins": 0, "Losses": 0, "Draws": 0})[k] += 1

    for A, B, o in zip(res["A"], res["B"], res["OUTCOME"]):
        if not isinstance(A, str) or not isinstance(B, str):
            continue
        if o == "W/L":
            bump(A, "Wins"); bump(B, "Losses")
        elif o == "L/W":
            bump(A, "Losses"); bump(B, "Wins")
        elif o == "D/D":
            bump(A, "Draws"); bump(B, "Draws")
    rec = pd.DataFrame.from_dict(rec, orient="index")

    # ---------- Fighter attributes (tale of the tape) ----------
    tott = tott.merge(det[["URL", "NICKNAME"]], on="URL", how="left")

    fight_names = set(res["A"].dropna()) | set(res["B"].dropna()) | set(st["FIGHTER"])
    name_map = build_name_map(fight_names, tott["FIGHTER"].unique())
    tott["FIGHTER"] = tott["FIGHTER"].replace(name_map)
    print(f"Matched {len(name_map)} profile names to fight-card spellings "
          f"(e.g. {', '.join(f'{k} -> {v}' for k, v in list(name_map.items())[:3])})")

    # Fighters who appear in fights but have no profile row still get a row
    # (stats + record, no height/reach/age) so their fights aren't dropped.
    orphans = sorted(fight_names - set(tott["FIGHTER"]))
    if orphans:
        tott = pd.concat([tott, pd.DataFrame({"FIGHTER": orphans, "HEIGHT": "--", "WEIGHT": "--",
                                              "REACH": "--", "DOB": "--", "URL": "",
                                              "NICKNAME": np.nan})], ignore_index=True)
        print(f"{len(orphans)} fighters have fights but no profile: {', '.join(orphans[:10])}"
              + (" ..." if len(orphans) > 10 else ""))

    def completeness(r):
        return sum(str(r[c]) not in ("--", "nan") for c in ["HEIGHT", "WEIGHT", "REACH", "DOB"])

    tott["_score"] = tott.apply(completeness, axis=1)
    tott["_dob"] = pd.to_datetime(tott["DOB"], format="%b %d, %Y", errors="coerce")
    dup_names = sorted(tott.loc[tott["FIGHTER"].duplicated(keep=False), "FIGHTER"].unique())
    tott = (tott.sort_values(["FIGHTER", "_score", "_dob"], ascending=[True, False, False])
                .drop_duplicates("FIGHTER", keep="first"))

    f = pd.DataFrame({"Name": tott["FIGHTER"].values})
    f["Nickname"] = tott["NICKNAME"].fillna("No Name").replace("", "No Name").values
    # Missing weight/reach stay blank (NaN), not 0: a 0 turns into a fake
    # 70-inch reach difference, which badly skewed the model. Blank differences
    # are treated as "no difference" in training and prediction.
    f["Weight:"] = pd.to_numeric(tott["WEIGHT"].str.extract(r"(\d+)")[0], errors="coerce").values
    f["Age"] = ((today - tott["_dob"]).dt.days / 365.25).round(1).values
    f = f.set_index("Name")
    f = f.join(career, how="left")
    for c in career.columns:
        f[c] = f[c].fillna(0.0)
    f["Height:"] = tott.set_index("FIGHTER")["HEIGHT"].replace("--", np.nan)
    f["Reach:"] = pd.to_numeric(
        tott.set_index("FIGHTER")["REACH"].str.extract(r"(\d+)")[0], errors="coerce")

    # Manual height fixes for fighters ufcstats lists as "--"
    ov_path = os.path.join(HERE, "height_overrides.csv")
    if os.path.exists(ov_path):
        ov = pd.read_csv(ov_path, comment="#")
        applied = 0
        for _, r in ov.iterrows():
            n, inches = str(r["Name"]).strip(), r["HeightInches"]
            if n in f.index and pd.isna(f.at[n, "Height:"]) and pd.notna(inches):
                inches = int(inches)
                f.at[n, "Height:"] = f"{inches // 12}' {inches % 12}\""
                applied += 1
        print(f"Applied {applied} height overrides from height_overrides.csv")

    f = f.join(rec, how="left")
    for c in ["Wins", "Losses", "Draws"]:
        f[c] = f[c].fillna(0).astype(int)
    total = f["Wins"] + f["Losses"] + f["Draws"]
    f["Win Percentage"] = np.where(total > 0, (f["Wins"] / total.where(total > 0, 1)).round(2), 0.0)

    cols = ["Nickname", "Weight:", "Age", "SLpM:", "Str. Acc.:", "SApM:", "Str. Def:",
            "TD Avg.:", "TD Acc.:", "TD Def.:", "Sub. Avg.:", "Height:", "Reach:",
            "Wins", "Losses", "Draws", "Win Percentage"]
    f_by_name = f.copy()
    dobs = tott.set_index("FIGHTER")["_dob"]
    f = f[cols].reset_index().rename(columns={"index": "Name"})

    # ---------- Matches ----------
    wl = res[res["OUTCOME"].isin(["W/L", "L/W"])].copy()
    wl["Winner"] = np.where(wl["OUTCOME"] == "W/L", wl["A"], wl["B"])
    wl["Loser"] = np.where(wl["OUTCOME"] == "W/L", wl["B"], wl["A"])
    flip = [stable_coin(str(u)) for u in wl["URL"]]
    wl["Red"] = np.where(flip, wl["Winner"], wl["Loser"])
    wl["Blue"] = np.where(flip, wl["Loser"], wl["Winner"])
    wl["Weight Class"] = wl["WEIGHTCLASS"].map(clean_weightclass)
    m = wl[["Red", "Blue", "Winner", "Weight Class"]]

    # ---------- Point-in-time training rows ----------
    res_pit = res.join(wl[["Red", "Blue", "Winner"]])
    pit = build_point_in_time(res_pit, per, ev, f_by_name, dobs)

    # ---------- Write ----------
    fp = os.path.join(a.out_dir, "fighters_clean.csv")
    mp = os.path.join(a.out_dir, "matches_clean.csv")
    tp = os.path.join(a.out_dir, "training_pit.csv")
    f.to_csv(fp, index=False)
    m.to_csv(mp, index=False)
    pit.to_csv(tp, index=False)
    print(f"Wrote {len(pit)} point-in-time training rows -> {tp} "
          f"({pit['date'].min().date()} to {pit['date'].max().date()})")

    in_db = m["Red"].isin(f["Name"]) & m["Blue"].isin(f["Name"])
    print(f"Wrote {len(f)} fighters -> {fp}")
    print(f"Wrote {len(m)} decided fights -> {mp} "
          f"({in_db.sum()} with both fighters in DB; red-corner win rate {(m.Winner == m.Red).mean():.1%})")
    print(f"Fighters with no height: {f['Height:'].isna().sum()} | with UFC fights: {(total > 0).sum()}")
    if dup_names:
        print(f"Note: {len(dup_names)} names belong to more than one fighter; kept the most complete "
              f"profile and pooled their fight stats: {', '.join(dup_names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
