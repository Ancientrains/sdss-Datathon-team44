import pandas as pd
import pgeocode
import folium
import numpy as np
import math
from pathlib import Path

postal_codes = [
    "M9W1J1",
    "M9W1J1",
    "M5S2P1",
    "M2J4R1",
    "M2J4R1",
    "M9W6P8",
    "M6R2K3",
]

# 1) Clean and collapse only exact postal-code duplicates
s = pd.Series(postal_codes, name="postal_code").str.replace(" ", "").str.upper()

df = s.value_counts().rename_axis("postal_code").reset_index(name="count")

# pgeocode's Canada dataset resolves the 3-character FSA reliably.
df["lookup_postal_code"] = df["postal_code"].str[:3]

# 2) Geocode using pgeocode (Canada dataset)
nomi = pgeocode.Nominatim("ca")
geo = nomi.query_postal_code(df["lookup_postal_code"].tolist())

# pgeocode returns a DataFrame-like object; merge results
df["lat"] = geo["latitude"].to_numpy()
df["lon"] = geo["longitude"].to_numpy()

# 3) Drop failures (unmatched codes)
df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

if df.empty:
    raise ValueError("No valid postal codes could be geocoded.")

# 4) Make an interactive map centered on mean location
center = [df["lat"].mean(), df["lon"].mean()]
m = folium.Map(location=center, zoom_start=11)

# 15 miles in meters
RADIUS_METERS = 2 * 1609.344

for _, row in df.iterrows():
    popup = f'{row["postal_code"]} (n={row["count"]})'
    folium.Circle(
        location=[row["lat"], row["lon"]],
        radius=RADIUS_METERS,
        popup=popup,
        color="#3388ff",
        weight=1,
        fill=True,
        fill_color="#3388ff",
        fill_opacity=0.2,   # low per-layer opacity — overlapping areas accumulate naturally
        opacity=0.4,
    ).add_to(m)

output_path = Path(__file__).with_name("map.html")
m.save(output_path)
print(f"Map saved to: {output_path}")
