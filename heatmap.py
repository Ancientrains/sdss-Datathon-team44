from pathlib import Path
import hashlib
import json
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import pgeocode

# -------------------------
# Files
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
CSV_FILE  = BASE_DIR / "Public_services.csv"
CSV_FILE_FALLBACK = BASE_DIR / "Public_services_pressure.csv"
GEO_FILE  = BASE_DIR / "lfsa000b16a_e.shp"
MODEL_FILE = BASE_DIR / "pressure_model.pkl"
SIMULATED_CSV_FILE = BASE_DIR / "simulated_pressure_data_2025-12-31_to_2026-03-30.csv"

if not CSV_FILE.exists():
    CSV_FILE = CSV_FILE_FALLBACK

assert CSV_FILE.exists(), f"Missing source CSV. Checked: {BASE_DIR / 'Public_services.csv'} and {CSV_FILE_FALLBACK}"
assert MODEL_FILE.exists(), f"Missing: {MODEL_FILE}"

# -------------------------
# Column names
# -------------------------
POST_COL = "LOCATION_POSTAL_CODE"
DATE_COL = "OCCUPANCY_DATE"
CAP_COL  = "ACTUAL_CAPACITY"
OCC_COL  = "OCCUPIED_CAPACITY"
TARGET_COL = "PRESSURE_SCORE_GAUSSIAN"
CAT_FEATURES = [
    "LOCATION_POSTAL_CODE",
    "SECTOR",
    "OVERNIGHT_SERVICE_TYPE",
    "PROGRAM_MODEL",
    "PROGRAM_AREA",
    "CAPACITY_TYPE",
]
NUM_FEATURES = [CAP_COL, "lat", "lon", "dow", "month", "day"]
# -------------------------
# Variable define
# -------------------------
CIRCLE_RADIUS_M = 100          # blue circle radius (metres)
GREEN_MIN_M     = 0.75 * 1609.344  # green min radius (0.75 miles)
GREEN_MAX_M     = 2.0  * 1609.344  # green max radius (2 miles)
PULSE_OVERSHOOT = 0.35         # animation variables
ANIM_MS         = 650          # animation variables durations
STEP_MS         = 1400         # auto-play
SIMULATION_CUTOFF = pd.Timestamp("2025-12-30")
SIMULATION_MONTHS = 6
SIMULATION_LOOKBACK_WEEKS = 4


def load_model_bundle(model_path: Path) -> dict:
    with model_path.open("rb") as fh:
        bundle = pickle.load(fh)

    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"Unexpected model bundle format in {model_path}.")

    return bundle


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def build_simulation_seed(
    actual_df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    lookback_weeks: int,
    cat_features: list[str],
) -> pd.DataFrame:
    recent_start = cutoff_date - pd.Timedelta(weeks=lookback_weeks)
    recent = actual_df[
        (actual_df[DATE_COL] <= cutoff_date) & (actual_df[DATE_COL] >= recent_start)
    ].copy()
    if recent.empty:
        raise ValueError("No historical shelter rows available for simulation seeding.")

    recent["week_start"] = recent[DATE_COL].dt.to_period("W").dt.start_time
    weekly_caps = (
        recent.groupby(cat_features + ["week_start"], dropna=False)[CAP_COL]
        .mean()
        .reset_index()
    )

    trend_rows = []
    for key, group in weekly_caps.groupby(cat_features, dropna=False):
        ordered = group.sort_values("week_start")
        caps = ordered[CAP_COL].to_numpy(dtype=float)
        last_capacity = float(caps[-1])
        slope_per_week = 0.0
        if len(caps) > 1:
            slope_per_week = float(np.polyfit(np.arange(len(caps), dtype=float), caps, 1)[0])
            slope_cap = max(1.0, abs(last_capacity) * 0.25)
            slope_per_week = float(np.clip(slope_per_week, -slope_cap, slope_cap))

        row = dict(zip(cat_features, key if isinstance(key, tuple) else (key,), strict=False))
        row["baseline_capacity"] = round(last_capacity, 2)
        row["capacity_slope_per_week"] = round(slope_per_week, 4)
        row["history_weeks_used"] = int(len(caps))
        trend_rows.append(row)

    trend_df = pd.DataFrame(trend_rows)
    base_df = (
        recent.sort_values(DATE_COL)
        .groupby(cat_features, dropna=False)
        .agg(
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            last_observed_date=(DATE_COL, "max"),
            baseline_occupied_capacity=(OCC_COL, "mean"),
        )
        .reset_index()
    )
    seed_df = base_df.merge(trend_df, on=cat_features, how="inner")
    seed_df["baseline_occupied_capacity"] = seed_df["baseline_occupied_capacity"].round(2)
    seed_df["baseline_occupancy_rate"] = (
        seed_df["baseline_occupied_capacity"] / seed_df["baseline_capacity"].replace({0: np.nan})
    ).clip(lower=0)

    return seed_df


def build_simulated_frame(
    seed_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    future_dates = pd.date_range(start_date, end_date, freq="D")
    if future_dates.empty:
        raise ValueError("Simulation date range is empty.")

    future = seed_df.assign(_join_key=1).merge(
        pd.DataFrame({DATE_COL: future_dates, "_join_key": 1}),
        on="_join_key",
        how="inner",
    )
    future = future.drop(columns="_join_key")
    future["week_offset"] = ((future[DATE_COL] - start_date).dt.days // 7).astype(float)
    future[CAP_COL] = (
        future["baseline_capacity"] + future["capacity_slope_per_week"] * future["week_offset"]
    ).clip(lower=0.0).round(2)
    future["dow"] = future[DATE_COL].dt.dayofweek
    future["month"] = future[DATE_COL].dt.month
    future["day"] = future[DATE_COL].dt.day

    return future


def generate_simulated_pressure_data(actual_df: pd.DataFrame) -> pd.DataFrame:
    bundle = load_model_bundle(MODEL_FILE)
    model = bundle["model"]
    cat_features = bundle.get("cat_features", CAT_FEATURES)
    num_features = bundle.get("num_features", NUM_FEATURES)

    actual_history = actual_df[actual_df[DATE_COL] <= SIMULATION_CUTOFF].copy()
    seed_df = build_simulation_seed(
        actual_df=actual_history,
        cutoff_date=SIMULATION_CUTOFF,
        lookback_weeks=SIMULATION_LOOKBACK_WEEKS,
        cat_features=cat_features,
    )

    forecast_start = SIMULATION_CUTOFF + pd.Timedelta(days=1)
    forecast_end = SIMULATION_CUTOFF + pd.DateOffset(months=SIMULATION_MONTHS)
    simulated_df = build_simulated_frame(seed_df, forecast_start, forecast_end)
    raw_predictions = model.predict(simulated_df[cat_features + num_features])
    simulated_df[TARGET_COL] = sigmoid(raw_predictions).round(6)
    simulated_df["DATA_SOURCE"] = "simulated"
    simulated_df["SIMULATION_CUTOFF_DATE"] = SIMULATION_CUTOFF.normalize()

    output_cols = [
        DATE_COL,
        "DATA_SOURCE",
        "SIMULATION_CUTOFF_DATE",
        "last_observed_date",
        "history_weeks_used",
        "capacity_slope_per_week",
        "baseline_capacity",
        "baseline_occupied_capacity",
        "baseline_occupancy_rate",
        *cat_features,
        CAP_COL,
        "lat",
        "lon",
        "dow",
        "month",
        "day",
        TARGET_COL,
    ]
    simulated_df = simulated_df[output_cols].sort_values([DATE_COL, POST_COL]).reset_index(drop=True)

    return simulated_df


# -------------------------
# 1) Load + clean
# -------------------------
df = pd.read_csv(CSV_FILE, usecols=[DATE_COL, *CAT_FEATURES, CAP_COL, OCC_COL])

df[POST_COL] = df[POST_COL].astype("string")
df = df.dropna(subset=[POST_COL]).copy()
df[POST_COL] = df[POST_COL].str.upper().str.replace(" ", "", regex=False)

pattern = r"[ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z]\d[ABCEGHJ-NPRSTV-Z]\d"
df = df[df[POST_COL].str.fullmatch(pattern, na=False)].copy()
df["FSA"] = df[POST_COL].str[:3]

df[CAP_COL] = pd.to_numeric(df[CAP_COL], errors="coerce")
df[OCC_COL] = pd.to_numeric(df[OCC_COL], errors="coerce")
df = df.dropna(subset=[CAP_COL, OCC_COL]).copy()
df["OCCUPANCY_RATE"] = df[OCC_COL] / df[CAP_COL].replace({0: np.nan})

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL]).copy()

# -------------------------
# 2) Geocode postal codes via pgeocode (with FSA fallback)
# -------------------------
unique_codes = df[POST_COL].unique().tolist()
nomi = pgeocode.Nominatim("ca")

geo_full = nomi.query_postal_code(unique_codes)
geo_map = pd.DataFrame({
    "postal_code": unique_codes,
    "lat": geo_full["latitude"].to_numpy(),
    "lon": geo_full["longitude"].to_numpy(),
})

missing_mask = geo_map["lat"].isna() | geo_map["lon"].isna()
if missing_mask.any():
    fsas = geo_map.loc[missing_mask, "postal_code"].str[:3]
    geo_fsa = nomi.query_postal_code(fsas.tolist())
    geo_map.loc[missing_mask, "lat"] = geo_fsa["latitude"].to_numpy()
    geo_map.loc[missing_mask, "lon"] = geo_fsa["longitude"].to_numpy()

df = df.merge(geo_map.rename(columns={"postal_code": POST_COL}), on=POST_COL, how="left")
df = df.dropna(subset=["lat", "lon"]).copy()

# Deterministic jitter for duplicate coords
dupe = df.duplicated(subset=["lat", "lon", POST_COL], keep=False)
if dupe.any():
    def jitter(code: str) -> tuple[float, float]:
        h = int(hashlib.md5(code.encode()).hexdigest()[:8], 16)
        return ((h % 1000) / 999 - 0.5) * 0.004, (((h // 1000) % 1000) / 999 - 0.5) * 0.004
    offsets = df.loc[dupe, POST_COL].map(jitter)
    df.loc[dupe, "lat"] += offsets.map(lambda t: t[0]).to_numpy()
    df.loc[dupe, "lon"] += offsets.map(lambda t: t[1]).to_numpy()

simulated_df = generate_simulated_pressure_data(df)
simulated_df.to_csv(SIMULATED_CSV_FILE, index=False)
print(
    "Simulated data saved:",
    SIMULATED_CSV_FILE,
    f"({len(simulated_df)} rows from {simulated_df[DATE_COL].min().date()} to {simulated_df[DATE_COL].max().date()})",
)

# -------------------------
# 3) Weekly binning
# -------------------------
df["BIN"] = df[DATE_COL].dt.to_period("W").astype(str)
bins_all = sorted(df["BIN"].unique().tolist())

start_period = pd.Period("2024-01-01", freq="W")
bins = sorted([b for b in bins_all if pd.Period(b, freq="W") >= start_period])
if not bins:
    raise ValueError("No weekly bins found after start_period.")

print(f"Weekly bins: {bins[0]} … {bins[-1]}  ({len(bins)} weeks)")

# Aggregate per week × postal_code
agg = (
    df.groupby(["BIN", POST_COL, "lat", "lon"])
      .agg(total_cap=(CAP_COL, "mean"), occupied=(OCC_COL, "mean"))
      .reset_index()
)
agg["rate"] = agg["occupied"] / agg["total_cap"].replace({0: np.nan})
agg = agg.dropna(subset=["rate"]).copy()
agg = agg[agg["BIN"].isin(bins)].copy()

# -------------------------
# 4) Radius scaling by total capacity (metres)
# -------------------------
CAP_MIN_R = 80
CAP_MAX_R = 500
cap_min = float(agg["total_cap"].min())
cap_max = float(agg["total_cap"].max())
if cap_max <= cap_min:
    cap_max = cap_min + 1

def cap_radius(cap: float) -> float:
    x = max(0.0, min(1.0, (float(cap) - cap_min) / (cap_max - cap_min)))
    return round(CAP_MIN_R + x * (CAP_MAX_R - CAP_MIN_R), 1)

agg["radius_m"] = agg["total_cap"].apply(cap_radius)

def green_radius(cap: float) -> float:
    x = max(0.0, min(1.0, (float(cap) - cap_min) / (cap_max - cap_min)))
    return round(GREEN_MIN_M + x * (GREEN_MAX_M - GREEN_MIN_M), 1)

agg["green_r"] = agg["total_cap"].apply(green_radius)

# -------------------------
# 5) Load + bin simulated data
# -------------------------
sim_agg = pd.DataFrame()
if SIMULATED_CSV_FILE.exists():
    sim_df = pd.read_csv(SIMULATED_CSV_FILE)
    sim_df[DATE_COL] = pd.to_datetime(sim_df[DATE_COL], errors="coerce")
    sim_df = sim_df.dropna(subset=[DATE_COL, "lat", "lon", TARGET_COL]).copy()
    sim_df[POST_COL] = sim_df[POST_COL].astype("string").str.upper().str.replace(" ", "", regex=False)

    sim_df["BIN"] = sim_df[DATE_COL].dt.to_period("W").astype(str)
    sim_agg = (
        sim_df.groupby(["BIN", POST_COL, "lat", "lon"])
              .agg(total_cap=(CAP_COL, "mean"), rate=(TARGET_COL, "mean"))
              .reset_index()
    )
    sim_agg = sim_agg.dropna(subset=["rate"]).copy()
    # Use same cap range as actual data for consistent radius scaling
    sim_agg["radius_m"] = sim_agg["total_cap"].apply(cap_radius)
    sim_agg["green_r"]  = sim_agg["total_cap"].apply(green_radius)
    print(f"Simulated bins: {sim_agg['BIN'].min()} … {sim_agg['BIN'].max()}  ({sim_agg['BIN'].nunique()} weeks)")
else:
    print(f"Warning: simulated CSV not found at {SIMULATED_CSV_FILE}. Skipping simulated data.")

# -------------------------
# 6) Build per-shelter weekly timeline dict for JS
#    Each entry carries a "sim" flag so JS can colour it differently
# -------------------------
shelter_data: dict = {}

# Actual data
for _, row in agg.iterrows():
    sid = row[POST_COL]
    if sid not in shelter_data:
        shelter_data[sid] = {}
    shelter_data[sid][row["BIN"]] = {
        "lat":      round(float(row["lat"]), 6),
        "lon":      round(float(row["lon"]), 6),
        "rate":     round(float(row["rate"]), 4),
        "radius_m": float(row["radius_m"]),
        "green_r":  float(row["green_r"]),
        "sim":      False,
    }

# Simulated data — merged in after, so it extends the timeline
for _, row in sim_agg.iterrows():
    sid = row[POST_COL]
    if sid not in shelter_data:
        shelter_data[sid] = {}
    shelter_data[sid][row["BIN"]] = {
        "lat":      round(float(row["lat"]), 6),
        "lon":      round(float(row["lon"]), 6),
        "rate":     round(float(row["rate"]), 4),
        "radius_m": float(row["radius_m"]),
        "green_r":  float(row["green_r"]),
        "sim":      True,
    }

# Merge actual + simulated bin lists into one sorted timeline
all_bins = sorted(set(bins) | set(sim_agg["BIN"].unique().tolist() if not sim_agg.empty else []))

# -------------------------
# 7) Build base map
# -------------------------
center_lat = float(agg["lat"].mean())
center_lon = float(agg["lon"].mean())

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=11,
    tiles="CartoDB positron",
    control_scale=True,
)


# -------------------------
# 8) Choropleth (static background, Blues palette)
# -------------------------
if GEO_FILE.exists():
    gdf = gpd.read_file(GEO_FILE)
    FSA_COL = "CFSAUID"
    toronto_fsa = gdf[gdf[FSA_COL].astype(str).str.startswith("M")][[FSA_COL, "geometry"]].copy()
    toronto_fsa["geometry"] = toronto_fsa["geometry"].simplify(0.001, preserve_topology=True)
    toronto_fsa = toronto_fsa.to_crs(epsg=4326)

    fsa_counts = (
        df.groupby("FSA")[POST_COL].count()
          .reset_index().rename(columns={POST_COL: "count"})
    )
    map_df = toronto_fsa.merge(fsa_counts, left_on=FSA_COL, right_on="FSA", how="left")
    map_df["count"] = map_df["count"].fillna(0)

    folium.Choropleth(
        geo_data=map_df, data=map_df,
        columns=[FSA_COL, "count"],
        key_on=f"feature.properties.{FSA_COL}",
        fill_color="Blues", fill_opacity=0.7, line_opacity=0.3,
        legend_name="Count of Services",
    ).add_to(m)
else:
    print(f"Warning: shapefile not found at {GEO_FILE}. Skipping choropleth.")

# -------------------------
# 9) Inject animated weekly timeline via custom Leaflet JS
#    Circles keep their original fixed colours and fixed radii at all times.
#    When occupancy_rate changes week-over-week (> 1 pp), a pulse fires:
#      both circles overshoot by PULSE_OVERSHOOT then ease back to their
#      fixed radii — same behaviour whether rate went up or down.
# -------------------------


shelter_json = json.dumps(shelter_data)
bins_json    = json.dumps(all_bins)

# Folium names its JS map variable "map_<id>" — inject it directly so
# the script never has to guess by scanning window properties.
map_var = f"map_{m._id}"

js = f"""
<script>
(function() {{

// ── Data ─────────────────────────────────────────────────────────────────────
var SHELTER_DATA = {shelter_json};
var BINS         = {bins_json};
var OVERSHOOT    = {PULSE_OVERSHOOT};
var ANIM_MS      = {ANIM_MS};
var STEP_MS      = {STEP_MS};
var BLUE_R       = {CIRCLE_RADIUS_M};
// green halo radius is per-shelter via d.green_r

// ── State ─────────────────────────────────────────────────────────────────────
var currentBin = 0;
var playing    = false;
var playTimer  = null;
var circles    = {{}};  // sid → {{ blue: L.circle, green: L.circle }}
var prevRate   = {{}};  // sid → last week's occupancy rate
var showActualHalo = true;
var showSimHalo    = true;

// ── Colour scale ─────────────────────────────────────────────────────────────
// Actual data palette (blue → orange)
var PALETTE = ["#386D9C","#4C6D8D","#606C7D","#9B6A4E","#AF693F","#C3682F","#D7671F","#EB6710","#FF6600"];
// Simulated data palette (purple tones — visually distinct from actual)
var SIM_PALETTE = ["#7B5EA7","#8B6DB5","#9B7CC3","#7A5F9A","#8C5F9E","#9E5FA2","#B05FA6","#C25FAA","#D45FAE"];
var SIM_HALO    = "#7B5EA7";  // simulated halo colour

function rateColor(rate, isSim) {{
    var pal = isSim ? SIM_PALETTE : PALETTE;
    var lowColor = isSim ? "#6B8F71" : "#2e8b57";   // muted sage vs vivid green
    var midColor = isSim ? "#7B5EA7" : "#386D9C";   // purple vs blue
    if (rate < 0.5) return lowColor;
    if (rate < 0.8) return midColor;
    var snapped = Math.round(Math.min((rate - 0.8) / 0.2, 1.0) * (pal.length - 1));
    return pal[Math.min(snapped, pal.length - 1)];
}}

function haloVisible(isSim) {{
    return isSim ? showSimHalo : showActualHalo;
}}

function haloStyle(haloCol, isSim) {{
    var visible = haloVisible(isSim);
    return {{
        color: haloCol,
        fillColor: haloCol,
        opacity: visible ? 0.28 : 0.0,
        fillOpacity: visible ? 0.12 : 0.0
    }};
}}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
function init() {{
    // Use the map variable name Folium generates — no window scanning needed
    var MAP = window["{map_var}"];
    if (!MAP) {{ setTimeout(init, 100); return; }}

    // Panes to control draw order (green under blue)
    var haloPane = MAP.createPane("halo-pane");
    haloPane.style.zIndex = 250;
    haloPane.style.pointerEvents = "none";
    var dotPane = MAP.createPane("dot-pane");
    dotPane.style.zIndex = 450;

    for (var sid in SHELTER_DATA) {{
        var weeks    = SHELTER_DATA[sid];
        var firstBin = Object.keys(weeks)[0];
        var d        = weeks[firstBin];

        var haloCol = d.sim ? SIM_HALO : "#2e8b57";
        var green = L.circle([d.lat, d.lon], {{
            radius:      d.green_r,
            color:       haloCol,
            weight:      1,
            fillColor:   haloCol,
            fillOpacity: haloVisible(d.sim) ? 0.12 : 0.0,
            opacity:     haloVisible(d.sim) ? 0.28 : 0.0,
            pane:        "halo-pane"
        }}).addTo(MAP);

        var col  = rateColor(d.rate, d.sim);
        var blue = L.circle([d.lat, d.lon], {{
            radius:      d.radius_m,
            color:       "#111",
            weight:      1.2,
            fillColor:   col,
            fillOpacity: 0.85,
            pane:        "dot-pane"
        }}).addTo(MAP).bindTooltip("", {{sticky: true}});

        blue.bringToFront();
        circles[sid] = {{ blue: blue, green: green }};
        prevRate[sid] = d.rate;
    }}

    renderBin(0);
    buildUI(MAP);
}}

// ── Ease-out cubic ────────────────────────────────────────────────────────────
function easeOut(t) {{ return 1 - Math.pow(1 - t, 3); }}

// ── Pulse: overshoot then settle back to target radius ───────────────────────
function pulse(blueCircle, targetR) {{
    var peakB  = targetR * (1 + OVERSHOOT);
    var PHASE1 = ANIM_MS * 0.40;
    var PHASE2 = ANIM_MS * 0.60;
    var start  = null;

    function step(ts) {{
        if (!start) start = ts;
        var el = ts - start;
        var t, rb, rg;

        if (el < PHASE1) {{
            t  = easeOut(el / PHASE1);
            rb = targetR + (peakB - targetR) * t;
        }} else if (el < PHASE1 + PHASE2) {{
            t  = easeOut((el - PHASE1) / PHASE2);
            rb = peakB + (targetR - peakB) * t;
        }} else {{
            rb = targetR;
        }}

        blueCircle.setRadius(rb);
        if (el < PHASE1 + PHASE2) requestAnimationFrame(step);
    }}
    requestAnimationFrame(step);
}}

// ── Render one week bin ───────────────────────────────────────────────────────
function renderBin(idx) {{
    currentBin = idx;
    var bin    = BINS[idx];

    var slider = document.getElementById("wk-slider");
    var label  = document.getElementById("wk-label");
    if (slider) slider.value = idx;
    if (label)  label.textContent = bin + (Object.values(SHELTER_DATA).some(function(w) {{ var e = w[bin]; return e && e.sim; }}) ? "  ★ forecast" : "");

    for (var sid in circles) {{
        var d = (SHELTER_DATA[sid] || {{}})[bin];
        if (!d) continue;

        var prev    = prevRate[sid] != null ? prevRate[sid] : d.rate;
        var col     = rateColor(d.rate, d.sim);
        var haloCol = d.sim ? SIM_HALO : "#2e8b57";
        var radius  = d.radius_m;

        circles[sid].blue.setLatLng([d.lat, d.lon]);
        circles[sid].green.setLatLng([d.lat, d.lon]);
        circles[sid].blue.setStyle({{ fillColor: col }});
        circles[sid].green.setStyle(haloStyle(haloCol, d.sim));
        circles[sid].blue.getTooltip().setContent(
            sid + " | " + Math.round(d.rate * 100) + "% occupancy"
        );

        // Pulse on any meaningful rate change (> 1 percentage point)
        circles[sid].green.setRadius(d.green_r);
        if (Math.abs(d.rate - prev) > 0.01) {{
            pulse(circles[sid].blue, radius);
        }} else {{
            circles[sid].blue.setRadius(radius);
        }}

        prevRate[sid] = d.rate;
    }}
}}

// ── Timeline UI ───────────────────────────────────────────────────────────────
function buildUI(MAP) {{
    var ctrl = L.control({{ position: "bottomright" }});
    ctrl.onAdd = function() {{
        var d = L.DomUtil.create("div");
        d.style.cssText = "background:rgba(0,0,0,0.75);color:#fff;padding:12px 16px;" +
            "border-radius:10px;font-family:sans-serif;font-size:13px;" +
            "min-width:290px;box-shadow:0 4px 16px rgba(0,0,0,.5);";
        d.innerHTML =
            '<div style="margin-bottom:6px;font-weight:700;font-size:14px;">Weekly Timeline</div>' +
            '<div style="display:flex;align-items:center;gap:8px;">' +
              '<button id="btn-prev" style="background:#333;color:#fff;border:none;' +
                'border-radius:4px;padding:3px 9px;cursor:pointer;font-size:14px;">◀</button>' +
              '<input id="wk-slider" type="range" min="0" max="' + (BINS.length - 1) + '" value="0"' +
                ' style="flex:1;accent-color:#1e64c8;">' +
              '<button id="btn-next" style="background:#333;color:#fff;border:none;' +
                'border-radius:4px;padding:3px 9px;cursor:pointer;font-size:14px;">▶</button>' +
            '</div>' +
            '<div style="display:flex;justify-content:space-between;align-items:center;margin-top:7px;">' +
              '<span id="wk-label" style="font-size:12px;color:#aaa;"></span>' +
              '<button id="btn-play" style="background:#1e64c8;color:#fff;border:none;' +
                'border-radius:4px;padding:4px 14px;cursor:pointer;">▶ Play</button>' +
            '</div>' +
            '<div style="display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:9px;font-size:12px;color:#ddd;">' +
              '<label style="display:flex;align-items:center;gap:6px;cursor:pointer;">' +
                '<input id="toggle-green-halo" type="checkbox" checked style="accent-color:#2e8b57;">' +
                '<span>Green Halo</span>' +
              '</label>' +
              '<label style="display:flex;align-items:center;gap:6px;cursor:pointer;">' +
                '<input id="toggle-purple-halo" type="checkbox" checked style="accent-color:#7B5EA7;">' +
                '<span>Purple Halo</span>' +
              '</label>' +
            '</div>';
        L.DomEvent.disableClickPropagation(d);
        L.DomEvent.disableScrollPropagation(d);
        return d;
    }};
    ctrl.addTo(MAP);

    setTimeout(function() {{
        document.getElementById("wk-slider").addEventListener("input", function() {{
            stopPlay(); renderBin(parseInt(this.value));
        }});
        document.getElementById("btn-prev").addEventListener("click", function() {{
            stopPlay(); renderBin(Math.max(0, currentBin - 1));
        }});
        document.getElementById("btn-next").addEventListener("click", function() {{
            stopPlay(); renderBin(Math.min(BINS.length - 1, currentBin + 1));
        }});
        document.getElementById("btn-play").addEventListener("click", function() {{
            playing ? stopPlay() : startPlay();
        }});
        document.getElementById("toggle-green-halo").addEventListener("change", function() {{
            showActualHalo = this.checked;
            renderBin(currentBin);
        }});
        document.getElementById("toggle-purple-halo").addEventListener("change", function() {{
            showSimHalo = this.checked;
            renderBin(currentBin);
        }});
    }}, 300);
}}

function startPlay() {{
    playing = true;
    document.getElementById("btn-play").textContent = "⏸ Pause";
    function tick() {{
        var next = currentBin + 1 >= BINS.length ? 0 : currentBin + 1;
        renderBin(next);
        playTimer = setTimeout(tick, STEP_MS);
    }}
    playTimer = setTimeout(tick, STEP_MS);
}}

function stopPlay() {{
    playing = false;
    clearTimeout(playTimer);
    var btn = document.getElementById("btn-play");
    if (btn) btn.textContent = "▶ Play";
}}

// Wait for DOM + Leaflet map to be ready
if (document.readyState === "loading") {{
    document.addEventListener("DOMContentLoaded", function() {{ setTimeout(init, 200); }});
}} else {{
    setTimeout(init, 200);
}}

}})();
</script>
"""

m.get_root().html.add_child(folium.Element(js))

# -------------------------
# 10) Save
# -------------------------
output_path = BASE_DIR / "map.html"
m.save(str(output_path))
print(f"Map saved to: {output_path}")
