from pathlib import Path
import hashlib
import json
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
GEO_FILE  = BASE_DIR / "lfsa000b16a_e.shp"

assert CSV_FILE.exists(), f"Missing: {CSV_FILE}"

# -------------------------
# Column names
# -------------------------
POST_COL = "LOCATION_POSTAL_CODE"
DATE_COL = "OCCUPANCY_DATE"
CAP_COL  = "ACTUAL_CAPACITY"
OCC_COL  = "OCCUPIED_CAPACITY"
# -------------------------
# Variable define
# -------------------------
CIRCLE_RADIUS_M = 100          # blue circle fixed radius (metres)
GREEN_MIN_M     = 0.75 * 1609.344  # green halo min radius (0.75 miles)
GREEN_MAX_M     = 2.0  * 1609.344  # green halo max radius (2 miles)
PULSE_OVERSHOOT = 0.35         # 35% overshoot on change
ANIM_MS         = 650          # pulse animation duration ms
STEP_MS         = 1400         # auto-play ms per week step
# -------------------------
# 1) Load + clean
# -------------------------
df = pd.read_csv(CSV_FILE, usecols=[POST_COL, DATE_COL, CAP_COL, OCC_COL])

df[POST_COL] = df[POST_COL].astype("string")
df = df.dropna(subset=[POST_COL]).copy()
df[POST_COL] = df[POST_COL].str.upper().str.replace(" ", "", regex=False)

pattern = r"[ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z]\d[ABCEGHJ-NPRSTV-Z]\d"
df = df[df[POST_COL].str.fullmatch(pattern, na=False)].copy()
df["FSA"] = df[POST_COL].str[:3]

df[CAP_COL] = pd.to_numeric(df[CAP_COL], errors="coerce")
df[OCC_COL] = pd.to_numeric(df[OCC_COL], errors="coerce")
df = df.dropna(subset=[CAP_COL, OCC_COL]).copy()

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
# 5) Build per-shelter weekly timeline dict for JS
# -------------------------
shelter_data: dict = {}

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
    }

# -------------------------
# 5) Build base map
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
# 6) Choropleth (static background, Blues palette)
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
# 7) Inject animated weekly timeline via custom Leaflet JS
#    Circles keep their original fixed colours and fixed radii at all times.
#    When occupancy_rate changes week-over-week (> 1 pp), a pulse fires:
#      both circles overshoot by PULSE_OVERSHOOT then ease back to their
#      fixed radii — same behaviour whether rate went up or down.
# -------------------------


shelter_json = json.dumps(shelter_data)
bins_json    = json.dumps(bins)

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

// ── Colour scale ─────────────────────────────────────────────────────────────
var PALETTE = ["#386D9C","#4C6D8D","#606C7D","#9B6A4E","#AF693F","#C3682F","#D7671F","#EB6710","#FF6600"];
function rateColor(rate) {{
    if (rate < 0.5) return "#2e8b57";
    if (rate < 0.8) return "#386D9C";
    var snapped = Math.round(Math.min((rate - 0.8) / 0.2, 1.0) * (PALETTE.length - 1));
    return PALETTE[Math.min(snapped, PALETTE.length - 1)];
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

        var green = L.circle([d.lat, d.lon], {{
            radius:      d.green_r,
            color:       "#2e8b57",
            weight:      1,
            fillColor:   "#2e8b57",
            fillOpacity: 0.12,
            opacity:     0.28,
            pane:        "halo-pane"
        }}).addTo(MAP);

        var col  = rateColor(d.rate);
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
    if (label)  label.textContent = bin;

    for (var sid in circles) {{
        var d = (SHELTER_DATA[sid] || {{}})[bin];
        if (!d) continue;

        var prev   = prevRate[sid] != null ? prevRate[sid] : d.rate;
        var col    = rateColor(d.rate);
        var radius = d.radius_m;

        circles[sid].blue.setLatLng([d.lat, d.lon]);
        circles[sid].green.setLatLng([d.lat, d.lon]);
        circles[sid].blue.setStyle({{ fillColor: col }});
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
# 8) Save
# -------------------------
output_path = BASE_DIR / "map.html"
m.save(str(output_path))
print(f"Map saved → {output_path}")
