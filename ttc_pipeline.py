"""
ttc_pipeline.py
================
End-to-end TTC Transit Service Reliability pipeline, all in one file:

    SECTION 1 — ETL: load raw TTC delay CSV/XLSX -> clean -> SQLite DB
    SECTION 2 — Feature engineering + baseline delay-prediction model

USAGE
-----
1. Drop your downloaded TTC delay file(s) into data/raw/
   (filenames containing "subway"/"bus"/"streetcar" are auto-tagged by mode)
   Expected columns (standard TTC Open Data schema, case-insensitive):
       Date, Time, Day, Station (or Location), Code,
       Min Delay, Min Gap, Bound, Line, Vehicle

2. (Optional) Drop an Environment Canada daily weather CSV into
   data/raw/weather.csv

3. Run:
       python ttc_pipeline.py            # runs ETL + model training
       python ttc_pipeline.py --etl-only # just builds the database
       python ttc_pipeline.py --model-only  # just trains (DB must exist)

If no raw files are found, a small SYNTHETIC sample is generated so the
pipeline can be smoke-tested end-to-end before real data is dropped in.
"""

import argparse
import glob
import os
import re
import sqlite3

import numpy as np
import pandas as pd

# ============================================================
# SECTION 0 — Config / paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DB_PATH = os.path.join(PROCESSED_DIR, "ttc_reliability.db")

MODE_KEYWORDS = {
    "subway": "Subway",
    "bus": "Bus",
    "streetcar": "Streetcar",
}

DELAY_THRESHOLD_MIN = 5  # a delay event >= 5 minutes counts as "delay-prone"


# ============================================================
# SECTION 1 — ETL: raw files -> cleaned SQLite database
# ============================================================

def guess_mode(filename: str) -> str:
    fname = filename.lower()
    for kw, label in MODE_KEYWORDS.items():
        if kw in fname:
            return label
    return "Subway"  # default assumption if filename doesn't say


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map many possible raw column-name variants to our canonical names."""
    rename_map = {}
    for col in df.columns:
        c = col.strip().lower().replace("_", " ")
        if c in ("date",):
            rename_map[col] = "date"
        elif c in ("time",):
            rename_map[col] = "time"
        elif c in ("day",):
            rename_map[col] = "day"
        elif c in ("station", "location", "stop"):
            rename_map[col] = "station"
        elif c in ("code",):
            rename_map[col] = "code"
        elif c in ("min delay", "mindelay"):
            rename_map[col] = "min_delay"
        elif c in ("min gap", "mingap"):
            rename_map[col] = "min_gap"
        elif c in ("bound",):
            rename_map[col] = "bound"
        elif c in ("line", "route"):
            rename_map[col] = "line"
        elif c in ("vehicle",):
            rename_map[col] = "vehicle"
    return df.rename(columns=rename_map)


def generate_synthetic_sample(n=15000, seed=42) -> pd.DataFrame:
    """
    Realistic-shaped SYNTHETIC placeholder data, clearly not real TTC
    records. Mirrors the known distribution shape (delay codes,
    rush-hour skew) purely so the pipeline is runnable before real data
    is dropped in. Replace by adding real files to data/raw/ and
    re-running.
    """
    rng = np.random.default_rng(seed)
    stations = [
        "BLOOR-YONGE STATION", "UNION STATION", "ST GEORGE STATION",
        "KENNEDY STATION", "FINCH STATION", "EGLINTON STATION",
        "DUNDAS STATION", "SHEPPARD-YONGE STATION", "SPADINA STATION",
        "KIPLING STATION",
    ]
    lines = ["YU", "BD", "SHP", "SRT"]
    codes = {
        "MUIS": "Miscellaneous Speed Control",
        "MUPAA": "Passenger Assistance Alarm",
        "MUATC": "ATC Project Related",
        "SUDP": "Disorderly Patron",
        "PUMEL": "Escalator/Elevator",
        "MUSC": "Signal Cutout",
        "TUSC": "No Operator Available",
    }
    dates = pd.date_range("2024-01-01", "2026-06-30", freq="D")
    date_choice = rng.choice(dates, size=n)
    hour_weights = np.array([1,1,1,1,1,2,4,8,9,6,3,3,3,3,4,6,9,10,7,4,3,2,1,1])
    hours = rng.choice(np.arange(24), size=n, p=hour_weights / hour_weights.sum())
    minutes = rng.integers(0, 60, size=n)

    df = pd.DataFrame({
        "date": pd.to_datetime(date_choice),
        "time": [f"{h:02d}:{m:02d}:00" for h, m in zip(hours, minutes)],
        "station": rng.choice(stations, size=n),
        "code": rng.choice(list(codes.keys()), size=n, p=[.22,.12,.08,.20,.15,.13,.10]),
        "min_delay": np.round(rng.gamma(shape=2.0, scale=4.0, size=n), 1),
        "min_gap": np.round(rng.gamma(shape=2.5, scale=5.0, size=n), 1),
        "bound": rng.choice(["N", "S", "E", "W"], size=n),
        "line": rng.choice(lines, size=n),
        "vehicle": rng.integers(5000, 6000, size=n).astype(str),
        "mode": "Subway",
    })
    df["day"] = df["date"].dt.day_name()
    return df


def load_delay_codes() -> pd.DataFrame:
    """
    Looks for a delay-codes lookup file (e.g. ttc-subway-delay-codes.xlsx)
    in data/raw/ and returns a (code, code_description) DataFrame.

    TTC's actual codes file has a messy layout: no clean single header
    row, often two side-by-side tables in one sheet (e.g. Subway codes
    in one block of columns, SRT codes in another), with the real
    header row several rows down and a row-number column mixed in.
    This reads the sheet with no assumed header, finds every column
    whose header cell contains "description" (case-insensitive), and
    treats the column immediately to its left as the matching code
    column — which is how TTC lays out each block.
    """
    candidates = [
        f for f in glob.glob(os.path.join(RAW_DIR, "*.xlsx")) + glob.glob(os.path.join(RAW_DIR, "*.xls"))
        if "code" in os.path.basename(f).lower()
    ]
    if not candidates:
        print("No delay-codes lookup file found in data/raw/ — code_description will stay blank.")
        return pd.DataFrame(columns=["code", "code_description"])

    f = candidates[0]
    print(f"Loading delay codes lookup from {os.path.basename(f)} ...")
    xls = pd.ExcelFile(f)
    all_pairs = []

    for sheet in xls.sheet_names:
        raw = xls.parse(sheet, header=None)

        # Find every cell that looks like a "description" header, anywhere
        # in the sheet, and pair it with the column directly to its left.
        for row_idx in range(min(10, len(raw))):  # header block is always near the top
            row = raw.iloc[row_idx]
            for col_idx, val in enumerate(row):
                if isinstance(val, str) and "description" in val.lower() and col_idx >= 1:
                    code_col, desc_col = col_idx - 1, col_idx
                    block = raw.iloc[row_idx + 1:, [code_col, desc_col]].copy()
                    block.columns = ["code", "code_description"]
                    block["code"] = block["code"].astype(str).str.strip().str.upper()
                    block["code_description"] = block["code_description"].astype(str).str.strip()
                    block = block[
                        block["code"].notna()
                        & (block["code"] != "NAN")
                        & (block["code"] != "")
                    ]
                    if len(block):
                        all_pairs.append(block)

    if not all_pairs:
        print(f"  Could not find code/description columns in {os.path.basename(f)} — leaving blank.")
        return pd.DataFrame(columns=["code", "code_description"])

    out = pd.concat(all_pairs, ignore_index=True).drop_duplicates(subset="code")
    print(f"  Found {len(out)} code descriptions across {len(all_pairs)} table block(s)")
    return out


def load_weather_data() -> pd.DataFrame:
    """
    Looks for Environment Canada daily weather CSV(s) in data/raw/
    (any filename containing "weather") and returns a combined
    (date_id, mean_temp_c, total_precip_mm, total_snow_cm) DataFrame.

    Environment Canada's export tool often only allows downloading one
    year at a time, so this combines ALL matching files (e.g.
    weather_2024.csv, weather_2025.csv, weather_2026.csv) rather than
    assuming a single file.

    Environment Canada's historical daily data export uses columns
    named "Date/Time", "Mean Temp (°C)", "Total Precip (mm)", and
    "Total Snow (cm)" — this matches on keywords rather than exact
    strings since the degree symbol and spacing can vary by export.
    Returns an empty DataFrame if no weather file is found.
    """
    candidates = [
        f for f in glob.glob(os.path.join(RAW_DIR, "*.csv"))
        if "weather" in os.path.basename(f).lower()
    ]
    if not candidates:
        print("No weather file found in data/raw/ — proceeding without weather features.")
        return pd.DataFrame(columns=["date_id", "mean_temp_c", "total_precip_mm", "total_snow_cm"])

    frames = []
    for f in sorted(candidates):
        print(f"Loading weather data from {os.path.basename(f)} ...")
        raw = pd.read_csv(f)

        col_map = {}
        for col in raw.columns:
            c = str(col).strip().lower()
            if "flag" in c:
                continue
            if "date" in c and "date_id" not in col_map.values():
                col_map[col] = "date_id"
            elif "mean temp" in c:
                col_map[col] = "mean_temp_c"
            elif "total precip" in c:
                col_map[col] = "total_precip_mm"
            elif "total snow" in c:
                col_map[col] = "total_snow_cm"

        if "date_id" not in col_map.values():
            print(f"  Could not find a date column in {os.path.basename(f)} — skipping this file.")
            continue

        weather = raw.rename(columns=col_map)
        keep_cols = [c for c in ["date_id", "mean_temp_c", "total_precip_mm", "total_snow_cm"] if c in weather.columns]
        weather = weather[keep_cols].copy()
        weather["date_id"] = pd.to_datetime(weather["date_id"], errors="coerce")
        weather = weather.dropna(subset=["date_id"])
        for c in ["mean_temp_c", "total_precip_mm", "total_snow_cm"]:
            if c in weather.columns:
                weather[c] = pd.to_numeric(weather[c], errors="coerce")
        frames.append(weather)

    if not frames:
        print("  No usable weather data found across matched files — proceeding without weather features.")
        return pd.DataFrame(columns=["date_id", "mean_temp_c", "total_precip_mm", "total_snow_cm"])

    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset="date_id")
    print(f"  Combined {len(candidates)} file(s) into {len(combined)} days of weather data")
    return combined


def load_raw_delay_files() -> pd.DataFrame:
    patterns = ["*.csv", "*.xlsx", "*.xls"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(RAW_DIR, p)))
    files = [
        f for f in files
        if "weather" not in os.path.basename(f).lower()
        and "code" not in os.path.basename(f).lower()
    ]

    if not files:
        print("No raw delay files found in data/raw/. Generating a synthetic sample instead.")
        return generate_synthetic_sample()

    frames = []
    for f in files:
        print(f"Loading {os.path.basename(f)} ...")
        mode = guess_mode(os.path.basename(f))
        if f.lower().endswith(".csv"):
            raw = pd.read_csv(f)
            raw = standardize_columns(raw)
            raw["mode"] = mode
            frames.append(raw)
        else:
            xls = pd.ExcelFile(f)
            for sheet in xls.sheet_names:
                sheet_df = xls.parse(sheet)
                sheet_df = standardize_columns(sheet_df)
                sheet_df["mode"] = mode
                frames.append(sheet_df)

    return pd.concat(frames, ignore_index=True, sort=False)


# TTC's raw delay data truncates the "Station" field at 22 characters
# at the source (a known quirk in the open dataset), and also logs
# between-station segments (e.g. "SPADINA TO WILSON STAT") and
# platform/direction modifiers (e.g. "(IN TUNNEL)", "- APPROACHING").
# This is the real list of TTC subway stations (Lines 1, 2, 4), used
# to map truncated/messy raw values back to a clean canonical name via
# longest-prefix matching, sorted longest-first so e.g. "EGLINTON WEST"
# is tried before "EGLINTON" and doesn't shadow it.
CANONICAL_STATIONS = [
    "VAUGHAN METROPOLITAN CENTRE", "HIGHWAY 407", "PIONEER VILLAGE", "YORK UNIVERSITY",
    "FINCH WEST", "DOWNSVIEW PARK", "SHEPPARD WEST", "WILSON", "YORKDALE",
    "LAWRENCE WEST", "GLENCAIRN", "EGLINTON WEST", "ST CLAIR WEST", "DUPONT",
    "SPADINA", "ST GEORGE", "MUSEUM", "QUEENS PARK", "ST PATRICK", "OSGOODE",
    "ST ANDREW", "UNION", "KING", "QUEEN", "DUNDAS WEST", "DUNDAS", "COLLEGE",
    "WELLESLEY", "BLOOR-YONGE", "ROSEDALE", "SUMMERHILL", "ST CLAIR", "DAVISVILLE",
    "EGLINTON", "LAWRENCE", "YORK MILLS", "SHEPPARD-YONGE", "NORTH YORK CENTRE", "FINCH",
    "KIPLING", "ISLINGTON", "ROYAL YORK", "OLD MILL", "JANE", "RUNNYMEDE",
    "HIGH PARK", "KEELE", "LANSDOWNE", "DUFFERIN", "OSSINGTON", "CHRISTIE",
    "BATHURST", "BAY", "SHERBOURNE", "CASTLE FRANK", "BROADVIEW", "CHESTER",
    "PAPE", "DONLANDS", "GREENWOOD", "COXWELL", "WOODBINE", "MAIN STREET",
    "VICTORIA PARK", "WARDEN", "KENNEDY", "BAYVIEW", "BESSARION", "LESLIE", "DON MILLS",
]
CANONICAL_STATIONS.sort(key=len, reverse=True)

STATION_ABBREVIATIONS = {
    "VMC": "VAUGHAN METROPOLITAN CENTRE",
    "VAUGHAN MC": "VAUGHAN METROPOLITAN CENTRE",
    "VAUGHAN METRO CENTRE": "VAUGHAN METROPOLITAN CENTRE",
    "NORTH YORK CTR": "NORTH YORK CENTRE",
    "N YORK CTR": "NORTH YORK CENTRE",
    "CEDARVALE": "EGLINTON WEST",  # Eglinton West station's newer name
    "KILPING": "KIPLING",           # known typo in the raw TTC data
    "LANDOWNE": "LANSDOWNE",        # known typo in the raw TTC data
    "WILSO": "WILSON",              # known typo in the raw TTC data
    "WISLON": "WILSON",             # known typo in the raw TTC data
}
# Sorted longest-first so e.g. "VAUGHAN METRO CENTRE" is tried before
# the shorter "VMC" would ever get a chance to (not actually a prefix
# collision here, but keeps the matching order predictable).
STATION_ABBREVIATIONS = dict(
    sorted(STATION_ABBREVIATIONS.items(), key=lambda kv: len(kv[0]), reverse=True)
)


def normalize_station_name(raw: str) -> str:
    """
    Maps a raw (possibly truncated/messy) TTC station field to a clean
    canonical station name, e.g. "CHESTER STATION - DEPA" -> "CHESTER
    STATION", "APPROACHING KIPLING" -> "KIPLING STATION", "563
    SHERBOURNE (SHERBO" -> "SHERBOURNE STATION".

    Real TTC data has the station name in inconsistent positions —
    sometimes first ("CHESTER STATION - DEPA"), sometimes after a
    modifier or numeric prefix ("APPROACHING KIPLING", "563
    SHERBOURNE"). Rather than assuming the station name starts the
    string, this searches for every canonical station name as a
    whole-word match anywhere in the text, and picks whichever one
    starts earliest (ties broken by longest name) — this preserves the
    "origin station" for between-station segments like "SPADINA TO ST
    GEORGE" (picks SPADINA, the earlier one) while still correctly
    handling modifier-first text.

    Returns "UNKNOWN / OTHER" for values that don't match any known
    station (e.g. pure line references like "SHEPPARD LINE").
    """
    if not isinstance(raw, str) or not raw.strip():
        return "UNKNOWN / OTHER"

    text = raw.strip().upper().replace(".", "")

    for abbr, full in STATION_ABBREVIATIONS.items():
        # Require the abbreviation to be a whole leading word/phrase
        # (followed by a space or end-of-string), not just any prefix —
        # otherwise a typo-fix like "WILSO" -> "WILSON" would wrongly
        # corrupt the already-correct "WILSON STATION" into
        # "WILSONN STATION".
        if text == abbr or text.startswith(abbr + " "):
            text = full + text[len(abbr):]
            break

    # Special-case: Bloor-Yonge is the one hyphenated interchange
    # station and shows up as "YONGE AND BLOOR", "YONGE BD STATION",
    # or standalone "YONGE STATION"/"BLOOR STATION" (bare "YONGE" or
    # "BLOOR" + STATION isn't a station on its own on Lines 1/2/4, so
    # it means the corresponding platform of the shared interchange).
    if ("BLOOR" in text and "YONGE" in text) or (
        text.startswith("YONGE") and ("STATION" in text or " BD" in text)
    ) or (
        text.startswith("BLOOR") and "STATION" in text and "WEST" not in text
    ):
        return "BLOOR-YONGE STATION"

    # Special-case: Sheppard-Yonge, same shared-interchange pattern.
    if ("SHEPPARD" in text and "YONGE" in text) or (
        text.startswith("SHEPPARD") and "STATION" in text and "WEST" not in text
    ):
        return "SHEPPARD-YONGE STATION"

    # "MAIN STATION" is shorthand for Main Street station.
    if text.startswith("MAIN STATION") or text.startswith("MAIN ST STATION"):
        return "MAIN STREET STATION"

    best_match = None  # (start_index, -len(name), name)
    for name in CANONICAL_STATIONS:
        m = re.search(r"\b" + re.escape(name) + r"\b", text)
        if m:
            candidate = (m.start(), -len(name), name)
            if best_match is None or candidate < best_match:
                best_match = candidate

    if best_match:
        return best_match[2] + " STATION"

    return "UNKNOWN / OTHER"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if "time" in df.columns:
        df["time"] = df["time"].astype(str).str.strip()
        # Extract the hour directly from the leading "HH:" — fast even on
        # 100K+ row real datasets, handles "HH:MM" and "HH:MM:SS" alike.
        df["hour"] = df["time"].str.extract(r"^(\d{1,2}):")[0].astype(float)
        df["hour"] = df["hour"].fillna(0).astype(int)
    else:
        df["time"] = "00:00:00"
        df["hour"] = 0

    for col in ("min_delay", "min_gap"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0.0

    before = len(df)
    df = df[~((df["min_delay"] == 0) & (df["min_gap"] == 0))]
    print(f"Dropped {before - len(df)} zero-delay/zero-gap non-events ({before} -> {len(df)} rows)")

    for col in ("station", "code", "bound", "line", "vehicle", "mode"):
        if col not in df.columns:
            df[col] = None
        else:
            df[col] = df[col].astype(str).str.strip().str.upper().replace({"NAN": None})

    # TTC's raw station field is truncated/messy at the source (see
    # normalize_station_name docstring) — map it to a clean canonical
    # station name and report the match rate so data quality is visible.
    raw_unique = df["station"].nunique()
    df["station"] = df["station"].apply(normalize_station_name)
    clean_unique = df["station"].nunique()
    unknown_count = (df["station"] == "UNKNOWN / OTHER").sum()
    unknown_pct = 100 * unknown_count / len(df) if len(df) else 0
    print(f"Station name cleanup: {raw_unique} raw variants -> {clean_unique} canonical stations "
          f"({unknown_count} rows / {unknown_pct:.1f}% unmatched as 'UNKNOWN / OTHER')")

    df["day"] = df["date"].dt.day_name()
    df["is_weekend"] = df["date"].dt.dayofweek >= 5

    month = df["date"].dt.month
    df["season"] = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
        ["Winter", "Spring", "Summer"],
        default="Fall",
    )

    return df


def build_database(df: pd.DataFrame):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)

    dim_date = (
        df[["date", "day", "is_weekend", "season"]]
        .drop_duplicates(subset="date")
        .rename(columns={"date": "date_id", "day": "day_of_week"})
    )
    dim_date["year"] = pd.to_datetime(dim_date["date_id"]).dt.year
    dim_date["month"] = pd.to_datetime(dim_date["date_id"]).dt.month
    dim_date["day"] = pd.to_datetime(dim_date["date_id"]).dt.day
    dim_date["is_holiday"] = False
    dim_date.to_sql("dim_date", conn, if_exists="replace", index=False)

    dim_station = (
        df[["station", "line"]].drop_duplicates(subset="station")
        .rename(columns={"station": "station_name"})
        .reset_index(drop=True)
    )
    dim_station.insert(0, "station_id", range(1, len(dim_station) + 1))
    dim_station.to_sql("dim_station", conn, if_exists="replace", index=False)
    station_map = dict(zip(dim_station["station_name"], dim_station["station_id"]))

    dim_code = df[["code"]].drop_duplicates()
    codes_lookup = load_delay_codes()
    dim_code = dim_code.merge(codes_lookup, on="code", how="left")
    if "category" not in dim_code.columns:
        dim_code["category"] = None
    dim_code.to_sql("dim_delay_code", conn, if_exists="replace", index=False)

    fact = df.copy()
    fact["station_id"] = fact["station"].map(station_map)
    fact = fact.rename(columns={"date": "date_id", "time": "time_of_day", "vehicle": "vehicle_id"})
    fact_cols = ["date_id", "time_of_day", "hour", "station_id", "code", "mode",
                 "bound", "line", "vehicle_id", "min_delay", "min_gap"]
    fact[fact_cols].to_sql("fact_delay_events", conn, if_exists="replace", index=False)

    weather = load_weather_data()
    if len(weather):
        weather.to_sql("fact_weather_daily", conn, if_exists="replace", index=False)
        print(f"  {len(weather)} days of weather data loaded into fact_weather_daily")

    conn.commit()
    conn.close()
    print(f"\nDatabase built: {DB_PATH}")
    print(f"  {len(dim_date)} dates, {len(dim_station)} stations, {len(fact)} delay events")


def run_etl():
    print("=" * 60)
    print("SECTION 1 — ETL")
    print("=" * 60)
    raw = load_raw_delay_files()
    cleaned = clean(raw)
    build_database(cleaned)


# ============================================================
# SECTION 2 — Feature engineering + baseline model
# ============================================================

def load_data_from_db() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    # Check whether fact_weather_daily has any rows before joining it in —
    # LEFT JOIN against an empty table is harmless but this keeps the
    # query and the "is weather available" logic in one obvious place.
    has_weather = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='fact_weather_daily'"
    ).fetchone()[0] > 0
    if has_weather:
        has_weather = conn.execute("SELECT COUNT(*) FROM fact_weather_daily").fetchone()[0] > 0

    weather_select = ""
    weather_join = ""
    if has_weather:
        weather_select = ", w.mean_temp_c, w.total_precip_mm, w.total_snow_cm"
        weather_join = "LEFT JOIN fact_weather_daily w ON f.date_id = w.date_id"

    query = f"""
        SELECT f.date_id, f.hour, f.min_delay, f.min_gap, f.code, f.mode,
               f.bound, f.line, s.station_name, d.day_of_week, d.is_weekend, d.season
               {weather_select}
        FROM fact_delay_events f
        LEFT JOIN dim_station s ON f.station_id = s.station_id
        LEFT JOIN dim_date d ON f.date_id = d.date_id
        {weather_join}
    """
    df = pd.read_sql(query, conn, parse_dates=["date_id"])
    conn.close()
    return df


def add_rolling_station_features(df: pd.DataFrame, windows=(7, 30)) -> pd.DataFrame:
    """
    Adds rolling delay-frequency features per station: for each event,
    how many delays did this station have in the preceding N days
    (NOT including the event's own day).

    Leakage safeguard: the rolling window is computed on a per-day
    delay count series, shifted forward by 1 day before the rolling
    sum, so a given event's own day never contributes to its own
    feature value — only strictly prior days do.
    """
    df = df.copy()
    df["date_only"] = pd.to_datetime(df["date_id"]).dt.normalize()

    daily = (
        df.groupby(["station_name", "date_only"])
        .size()
        .rename("daily_count")
        .reset_index()
    )

    all_dates = pd.date_range(daily["date_only"].min(), daily["date_only"].max(), freq="D")
    stations = daily["station_name"].dropna().unique()
    full_idx = pd.MultiIndex.from_product([stations, all_dates], names=["station_name", "date_only"])
    daily_full = (
        daily.set_index(["station_name", "date_only"])
        .reindex(full_idx, fill_value=0)
        .reset_index()
        .sort_values(["station_name", "date_only"])
    )

    roll_cols = []
    for w in windows:
        col = f"rolling_{w}d_station_delays"
        daily_full[col] = (
            daily_full.groupby("station_name")["daily_count"]
            .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).sum())
        )
        roll_cols.append(col)

    df = df.merge(daily_full[["station_name", "date_only"] + roll_cols], on=["station_name", "date_only"], how="left")
    for col in roll_cols:
        df[col] = df[col].fillna(0)

    return df


def engineer_features(df: pd.DataFrame):
    from sklearn.preprocessing import LabelEncoder

    df = df.copy()
    df["target"] = (df["min_delay"] >= DELAY_THRESHOLD_MIN).astype(int)

    cat_cols = ["hour", "day_of_week", "season", "station_name", "line", "bound", "mode"]
    df = df.dropna(subset=cat_cols)

    # Rolling per-station delay frequency (leakage-safe, see
    # add_rolling_station_features docstring). Computed here, before
    # the LabelEncoder loop, so it has access to the raw date/station
    # columns.
    df = add_rolling_station_features(df, windows=(7, 30))

    encoders = {}
    X = pd.DataFrame(index=df.index)
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    X["is_weekend"] = df["is_weekend"].astype(int)

    # Rush-hour flag: AM (7-9) and PM (16-18) peaks, based on the EDA's
    # own finding that these hours carry the highest delay *volume* even
    # though they're not the hours with the longest individual delays.
    X["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)

    # Weather features, if the weather table was populated during ETL
    # (see load_weather_data / fact_weather_daily). Left out entirely
    # if no weather data is available, rather than filling with zeros,
    # so the model isn't trained on fabricated weather signal.
    weather_cols = ["mean_temp_c", "total_precip_mm", "total_snow_cm"]
    available_weather_cols = [c for c in weather_cols if c in df.columns and df[c].notna().any()]
    for col in available_weather_cols:
        X[col] = df[col].fillna(df[col].median())

    for col in ("rolling_7d_station_delays", "rolling_30d_station_delays"):
        if col in df.columns:
            X[col] = df[col].fillna(0)

    y = df["target"]
    return X, y, encoders


def run_model_training():
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.model_selection import train_test_split
    import joblib

    print("\n" + "=" * 60)
    print("SECTION 2 — Feature engineering + model training")
    print("=" * 60)

    print("Loading data from SQLite...")
    df = load_data_from_db()
    print(f"  {len(df)} delay events loaded")

    print("Engineering features...")
    X, y, encoders = engineer_features(df)
    print(f"  Features used: {list(X.columns)}")
    print(f"  Target distribution: {y.value_counts(normalize=True).round(3).to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest kept as a quick comparison baseline — EDA showed GB
    # beats it by ~0.02 AUC on this feature set, small but real.
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    print("\nTraining GradientBoostingClassifier (primary model)...")
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    gb_auc = roc_auc_score(y_test, y_proba)

    print("\n--- Classification Report (Gradient Boosting) ---")
    print(classification_report(y_test, y_pred, target_names=["On-time-ish", "Delay-prone"]))
    print(f"Gradient Boosting ROC-AUC: {gb_auc:.3f}")
    print(f"Random Forest ROC-AUC:     {rf_auc:.3f}  (comparison baseline)")

    print("\n--- Feature Importances (Gradient Boosting) ---")
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(importances.round(3).to_string())

    joblib.dump(clf, os.path.join(PROCESSED_DIR, "delay_model.joblib"))
    joblib.dump(encoders, os.path.join(PROCESSED_DIR, "encoders.joblib"))
    print(f"\nModel + encoders saved to {PROCESSED_DIR}/")


def run_powerbi_export():
    """
    Exports clean, Power-BI-ready CSVs from the SQLite database into
    dashboard/powerbi_data/ — one CSV per table, matching the star
    schema (fact_delay_events, dim_station, dim_date, dim_delay_code,
    fact_weather_daily if present), plus a feature_importances.csv
    from the last trained model.

    Power BI doesn't read SQLite natively, so importing these CSVs via
    Get Data > Text/CSV and building relationships in Power BI (on
    station_id and date_id) mirrors a real star-schema BI workflow,
    rather than flattening everything into one denormalized table.
    """
    import joblib

    out_dir = os.path.join(BASE_DIR, "dashboard", "powerbi_data")
    os.makedirs(out_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    tables = ["fact_delay_events", "dim_station", "dim_date", "dim_delay_code"]
    has_weather = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='fact_weather_daily'"
    ).fetchone()[0] > 0
    if has_weather:
        tables.append("fact_weather_daily")

    for t in tables:
        df = pd.read_sql(f"SELECT * FROM {t}", conn)
        path = os.path.join(out_dir, f"{t}.csv")
        df.to_csv(path, index=False)
        print(f"  Exported {t}.csv ({len(df)} rows)")

    conn.close()

    model_path = os.path.join(PROCESSED_DIR, "delay_model.joblib")
    if os.path.exists(model_path):
        clf = joblib.load(model_path)
        df = load_data_from_db()
        X, y, encoders = engineer_features(df)
        importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
        importances_df = importances.reset_index()
        importances_df.columns = ["feature", "importance"]
        importances_df.to_csv(os.path.join(out_dir, "feature_importances.csv"), index=False)
        print(f"  Exported feature_importances.csv ({len(importances_df)} rows)")
    else:
        print("  No trained model found — skipping feature_importances.csv (run the model training step first)")

    print(f"\nAll Power BI CSVs written to: {out_dir}")


# ============================================================
# SECTION 3 — Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTC Transit Service Reliability pipeline")
    parser.add_argument("--etl-only", action="store_true", help="Only run the ETL step")
    parser.add_argument("--model-only", action="store_true", help="Only run model training (DB must already exist)")
    parser.add_argument("--powerbi-export", action="store_true", help="Export CSVs for Power BI to dashboard/powerbi_data/")
    args = parser.parse_args()

    if args.powerbi_export:
        run_powerbi_export()
    elif args.model_only:
        run_model_training()
    elif args.etl_only:
        run_etl()
    else:
        run_etl()
        run_model_training()
