"""
NEFTCHI ANALYTICS — Add New Match to CSV
────────────────────────────────────────
Usage:
    python add_match.py <file_path>
    python add_match.py data/match_data/Gabala_Neftchi_15_03.txt

Coordinate system (matches Cell 3a exactly):
    draw.start.x → pitch_x = sx × 68 / 100   (width,  0–68 m)
    draw.start.y → pitch_y = sy × 105 / 100   (depth,  0–105 m, goal at 0)
    No home/away flip needed — draw.start is already normalised.
"""

import pandas as pd
import json
import sys
from pathlib import Path
from datetime import datetime
import shutil

CSV_PATH = "data/neftchi_shots.csv"
BACKUP_FOLDER = "data/backups"


# ══════════════════════════════════════════════════════════════════════════════
# FILENAME PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_filename(filename):
    """
    Supported formats (matching Cell 3a logic):
        Opponent_Neftchi_DD_MM.txt   → away match
        Neftchi_Opponent_DD_MM.txt   → home match
        Multi_Word_Neftchi_DD_MM.txt → multi-word opponent
    """
    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) < 4:
        raise ValueError(
            f"Invalid filename: '{stem}'. "
            f"Expected: Opponent_Neftchi_DD_MM.txt"
        )

    day   = parts[-2]
    month = parts[-1]
    team_parts = parts[:-2]

    # Find "Neftchi" in the team parts (case-insensitive)
    try:
        neft_idx = [p.lower() for p in team_parts].index("neftchi")
    except ValueError:
        raise ValueError(
            f"'Neftchi' not found in filename: '{stem}'. "
            f"Team parts: {team_parts}"
        )

    if neft_idx == 0:
        home = "Neftchi"
        away = "_".join(team_parts[1:])
        neftchi_is_home = True
    else:
        home = "_".join(team_parts[:neft_idx])
        away = "Neftchi"
        neftchi_is_home = False

    rival = away if neftchi_is_home else home
    match_label = f"{home} - {away} ({day}.{month})"

    day_int   = int(day)
    month_int = int(month)

    # Season order: Aug(8)–Dec(12) = 2024, Jan(1)–Jul(7) = 2025
    # Same formula as Cell 3a
    season_order = (month_int if month_int >= 8 else month_int + 12) * 100 + day_int

    return {
        "match":           match_label,
        "rival":           rival,
        "day":             day_int,
        "month":           month_int,
        "date_str":        f"{day}.{month}",           # "15.03" — string, not float
        "season_order":    season_order,
        "neftchi_is_home": neftchi_is_home,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SHOT EXTRACTION  —  mirrors Cell 3a exactly
# ══════════════════════════════════════════════════════════════════════════════

def extract_neftchi_shots(json_data, match_info):
    """Extract Neftchi shots using draw.start coords (same as Cell 3a)."""
    shots = []

    for s in json_data["shotmap"]:
        # Only keep Neftchi's own shots
        if s.get("isHome") != match_info["neftchi_is_home"]:
            continue

        # ── Coordinates from draw.start (NOT playerCoordinates) ──────────
        draw  = s.get("draw", {})
        start = draw.get("start", {})
        sx    = start.get("x")
        sy    = start.get("y")
        if sx is None or sy is None:
            print(f"  ⚠ Skipping shot (no draw.start): {s.get('player', {}).get('shortName', '?')}")
            continue

        pitch_x = round(sx * 68  / 100, 2)    # width  0–68 m
        pitch_y = round(sy * 105 / 100, 2)     # depth  0–105 m  (goal at 0)

        # ── Player info ──────────────────────────────────────────────────
        player = s.get("player", {})
        player_name = player.get("shortName", player.get("name", "—"))
        jersey      = player.get("jerseyNumber", "—")
        position    = player.get("position", "—")

        # ── Raw labels (NO mapping — matches Cell 3a) ───────────────────
        shot_type = s.get("shotType", "miss")
        situation = s.get("situation", "regular")
        body_part = s.get("bodyPart", "right-foot")

        # ── Metrics ──────────────────────────────────────────────────────
        xg    = s.get("xg", 0.0)
        xgot  = s.get("xgot")
        xgot  = round(xgot, 4) if xgot is not None else None

        shots.append({
            "match":        match_info["match"],
            "rival":        match_info["rival"],
            "date":         match_info["date_str"],
            "season_order": match_info["season_order"],
            "player":       player_name,
            "jersey":       jersey,
            "position":     position,
            "shot_type":    shot_type,
            "situation":    situation,
            "body_part":    body_part,
            "xg":           round(xg, 4),
            "xgot":         xgot,
            "minute":       s.get("time", 0),
            "added":        s.get("addedTime", 0) or 0,
            "goal_mouth":   s.get("goalMouthLocation", ""),
            "pitch_x":      pitch_x,
            "pitch_y":      pitch_y,
        })

    return shots


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate(new_df, existing_df):
    issues = []

    # Check for nulls in critical columns
    for col in ["match", "date", "season_order", "player", "xg", "pitch_x", "pitch_y"]:
        if new_df[col].isnull().any():
            issues.append(f"⚠ Missing values in '{col}'")

    # Duplicate match check
    new_matches = set(new_df["match"].unique())
    old_matches = set(existing_df["match"].unique())
    dupes = new_matches & old_matches
    if dupes:
        issues.append(f"⚠ Match already exists in CSV: {dupes}")

    # Coordinate sanity (pitch_y up to ~40m covers 99% of shots)
    if (new_df["pitch_x"] < 0).any() or (new_df["pitch_x"] > 68).any():
        issues.append("⚠ pitch_x out of range (should be 0–68)")
    if (new_df["pitch_y"] < 0).any() or (new_df["pitch_y"] > 70).any():
        issues.append("⚠ pitch_y suspiciously large (>70m from goal)")

    # xG sanity
    if (new_df["xg"] < 0).any() or (new_df["xg"] > 1).any():
        issues.append("⚠ xG out of 0–1 range")

    return issues


# ══════════════════════════════════════════════════════════════════════════════
# BACKUP & SAVE
# ══════════════════════════════════════════════════════════════════════════════

def create_backup():
    Path(BACKUP_FOLDER).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = f"{BACKUP_FOLDER}/neftchi_shots_{ts}.csv"
    shutil.copy(CSV_PATH, dst)
    print(f"  💾 Backup → {dst}")
    return dst


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def add_match(file_path):
    print("═" * 60)
    print("  NEFTÇI PFK — Yeni Oyun Əlavə Et")
    print("═" * 60)

    # 1. Parse filename
    print(f"\n📂 Fayl: {file_path}")
    try:
        info = parse_filename(file_path)
    except ValueError as e:
        print(f"  ❌ {e}")
        return False

    venue = "HOME" if info["neftchi_is_home"] else "AWAY"
    print(f"  Oyun:         {info['match']}")
    print(f"  Rəqib:        {info['rival']}")
    print(f"  Tarix:        {info['day']}.{info['month']:02d}")
    print(f"  Yer:          {venue}")
    print(f"  season_order: {info['season_order']}")

    # 2. Load JSON
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        total = len(data.get("shotmap", []))
        print(f"\n  ✅ JSON yükləndi: {total} zərbə (hər iki komanda)")
    except Exception as e:
        print(f"  ❌ JSON xətası: {e}")
        return False

    # 3. Extract Neftchi shots
    rows = extract_neftchi_shots(data, info)
    print(f"  ✅ Neftçi zərbələri: {len(rows)}")

    if not rows:
        print("  ❌ Neftçi zərbəsi tapılmadı — isHome/away yoxlayın")
        return False

    df_new = pd.DataFrame(rows)

    # Quick summary
    goals = (df_new["shot_type"] == "goal").sum()
    xg_total = df_new["xg"].sum()
    print(f"  📊 {len(rows)} zərbə | {goals} qol | xG: {xg_total:.2f}")

    # 4. Load existing CSV
    try:
        df_old = pd.read_csv(CSV_PATH)
        print(f"\n  📁 Mövcud CSV: {len(df_old)} zərbə, {df_old['match'].nunique()} oyun")
    except FileNotFoundError:
        print(f"\n  ⚠ CSV tapılmadı, yeni yaradılır: {CSV_PATH}")
        df_old = pd.DataFrame()

    # 5. Validate
    print("\n  🔍 Yoxlama...")
    if len(df_old) > 0:
        issues = validate(df_new, df_old)
    else:
        issues = []

    if issues:
        print("  ⚠ PROBLEMLƏR:")
        for iss in issues:
            print(f"    {iss}")
        ans = input("\n  Davam etmək istəyirsiniz? (yes/no): ").strip().lower()
        if ans != "yes":
            print("  ❌ İmtina edildi")
            return False
    else:
        print("  ✅ Bütün yoxlamalar keçdi")

    # 6. Backup + Save
    if len(df_old) > 0:
        create_backup()

    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all = df_all.sort_values("season_order").reset_index(drop=True)

    try:
        df_all.to_csv(CSV_PATH, index=False)
    except Exception as e:
        print(f"  ❌ Saxlama xətası: {e}")
        return False

    # 7. Done
    print("\n" + "═" * 60)
    print("  ✅ UĞURLU!")
    print("═" * 60)
    print(f"  Oyun:       {info['match']}")
    print(f"  Əlavə:      {len(rows)} zərbə")
    print(f"  Yeni cəmi:  {len(df_all)} zərbə, {df_all['match'].nunique()} oyun")
    print(f"\n  → streamlit run app.py")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("İstifadə: python add_match.py <fayl_yolu>")
        print("Nümunə:   python add_match.py data/match_data/Gabala_Neftchi_15_03.txt")
        sys.exit(1)

    fp = sys.argv[1]
    if not Path(fp).exists():
        print(f"❌ Fayl tapılmadı: {fp}")
        sys.exit(1)

    ok = add_match(fp)
    sys.exit(0 if ok else 1)