# =============================================================================
# Toronto Shelter System - Data Analysis
# Three Research Questions:
#   Q1: Which sectors are most under pressure, and how consistently?
#   Q2: Does the season or time of year matter?
#   Q3: How much does unavailable capacity make things worse?
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# --- Plot styling ---
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 130
plt.rcParams['font.family'] = 'DejaVu Sans'
COLORS = sns.color_palette("Set2", 6)

# =============================================================================
# 0. LOAD & CLEAN DATA
# =============================================================================

print("Loading data...")
df = pd.read_excel('public_services_dataset.xlsx')

# Parse dates and extract time features
df['OCCUPANCY_DATE'] = pd.to_datetime(df['OCCUPANCY_DATE'])
df['YEAR']  = df['OCCUPANCY_DATE'].dt.year
df['MONTH'] = df['OCCUPANCY_DATE'].dt.month
df['MONTH_NAME'] = df['OCCUPANCY_DATE'].dt.strftime('%b')   # e.g. "Jan"
df['SEASON'] = df['MONTH'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring',  4: 'Spring', 5: 'Spring',
    6: 'Summer',  7: 'Summer', 8: 'Summer',
    9: 'Fall',   10: 'Fall',  11: 'Fall'
})

# Effective capacity = actual beds minus unavailable beds
# Clip at 1 to avoid division by zero
df['EFFECTIVE_CAPACITY'] = (df['ACTUAL_CAPACITY'] - df['UNAVAILABLE_CAPACITY']).clip(lower=1)

# Effective occupancy rate = occupied / effective capacity
df['EFFECTIVE_OCCUPANCY_RATE'] = (
    df['OCCUPIED_CAPACITY'] / df['EFFECTIVE_CAPACITY']
).clip(upper=1.5)   # cap extreme outliers for readability

# Flag days at or over full capacity (official rate)
df['AT_FULL_CAPACITY'] = df['OCCUPANCY_RATE'] >= 1.0

print(f"  Rows loaded   : {len(df):,}")
print(f"  Date range    : {df['OCCUPANCY_DATE'].min().date()} → {df['OCCUPANCY_DATE'].max().date()}")
print(f"  Sectors       : {df['SECTOR'].unique().tolist()}")
print()


# =============================================================================
# Q1: WHICH SECTORS ARE MOST UNDER PRESSURE, AND HOW CONSISTENTLY?
# =============================================================================
print("=" * 60)
print("Q1: Sector-Level Occupancy Pressure")
print("=" * 60)

sector_stats = (
    df.groupby('SECTOR')
    .agg(
        Avg_Occupancy_Rate   = ('OCCUPANCY_RATE',    'mean'),
        Pct_Days_Full        = ('AT_FULL_CAPACITY',  'mean'),   # proportion of days at 100%
        Total_Records        = ('OCCUPANCY_RATE',    'count'),
        Avg_Actual_Capacity  = ('ACTUAL_CAPACITY',   'mean'),
    )
    .reset_index()
    .sort_values('Avg_Occupancy_Rate', ascending=False)
)

sector_stats['Avg_Occupancy_Rate_Pct'] = sector_stats['Avg_Occupancy_Rate'] * 100
sector_stats['Pct_Days_Full_Pct']      = sector_stats['Pct_Days_Full']      * 100

print(sector_stats[['SECTOR','Avg_Occupancy_Rate_Pct','Pct_Days_Full_Pct','Avg_Actual_Capacity']].to_string(index=False))
print()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Q1: Sector Occupancy Pressure', fontsize=14, fontweight='bold', y=1.01)

# --- Chart A: Average occupancy rate by sector ---
ax = axes[0]
bars = ax.barh(
    sector_stats['SECTOR'],
    sector_stats['Avg_Occupancy_Rate_Pct'],
    color=COLORS[:len(sector_stats)]
)
ax.axvline(100, color='red', linestyle='--', linewidth=1.4, label='100% capacity')
ax.set_xlabel('Average Occupancy Rate (%)')
ax.set_title('Average Occupancy Rate by Sector')
ax.legend()
# Annotate bars
for bar, val in zip(bars, sector_stats['Avg_Occupancy_Rate_Pct']):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=9)
ax.set_xlim(0, 115)

# --- Chart B: % of days at full capacity ---
ax = axes[1]
bars = ax.barh(
    sector_stats['SECTOR'],
    sector_stats['Pct_Days_Full_Pct'],
    color=COLORS[:len(sector_stats)]
)
ax.axvline(50, color='orange', linestyle='--', linewidth=1.4, label='50% of days')
ax.set_xlabel('% of Days at 100% Occupancy')
ax.set_title('How Often Each Sector Hits Full Capacity')
ax.legend()
for bar, val in zip(bars, sector_stats['Pct_Days_Full_Pct']):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=9)
ax.set_xlim(0, 110)

plt.tight_layout()
plt.savefig('q1_sector_pressure.png', bbox_inches='tight')
#plt.show()
import os
os.makedirs("figures", exist_ok=True)

plt.tight_layout()
plt.savefig("figures/q1_sector_pressure.png", dpi=300)
plt.close()
print("  Saved: q1_sector_pressure.png\n")


# =============================================================================
# Q2: DOES THE SEASON OR TIME OF YEAR MATTER?
# =============================================================================
print("=" * 60)
print("Q2: Temporal Trends in Occupancy")
print("=" * 60)

# --- Monthly average occupancy rate (overall) ---
month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

monthly = (
    df.groupby(['YEAR', 'MONTH', 'MONTH_NAME'])
    ['OCCUPANCY_RATE'].mean()
    .reset_index()
)
monthly['MONTH_NAME'] = pd.Categorical(monthly['MONTH_NAME'], categories=month_order, ordered=True)
monthly = monthly.sort_values(['YEAR', 'MONTH'])

print("Monthly avg occupancy rate:")
print(monthly.pivot(index='MONTH_NAME', columns='YEAR', values='OCCUPANCY_RATE').round(3).to_string())
print()

# --- Seasonal average by sector ---
season_order = ['Winter', 'Spring', 'Summer', 'Fall']
season_sector = (
    df.groupby(['SEASON', 'SECTOR'])
    ['OCCUPANCY_RATE'].mean()
    .reset_index()
)
season_sector['SEASON'] = pd.Categorical(season_sector['SEASON'], categories=season_order, ordered=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Q2: Seasonal & Monthly Trends in Occupancy', fontsize=14, fontweight='bold', y=1.01)

# --- Chart A: Monthly trend by year ---
ax = axes[0]
for year, grp in monthly.groupby('YEAR'):
    ax.plot(grp['MONTH_NAME'], grp['OCCUPANCY_RATE'] * 100,
            marker='o', label=str(year), linewidth=2)
ax.axhline(100, color='red', linestyle='--', linewidth=1.2, label='100% capacity')
ax.set_xlabel('Month')
ax.set_ylabel('Avg Occupancy Rate (%)')
ax.set_title('Monthly Average Occupancy Rate (2024 vs 2025)')
ax.legend()
ax.set_ylim(88, 105)
plt.setp(ax.get_xticklabels(), rotation=45)

# --- Chart B: Seasonal breakdown by sector (heatmap) ---
ax = axes[1]
pivot = season_sector.pivot(index='SECTOR', columns='SEASON', values='OCCUPANCY_RATE') * 100
pivot = pivot[season_order]   # reorder columns
sns.heatmap(
    pivot,
    ax=ax,
    annot=True,
    fmt='.1f',
    cmap='YlOrRd',
    vmin=88, vmax=100,
    linewidths=0.5,
    cbar_kws={'label': 'Avg Occupancy Rate (%)'}
)
ax.set_title('Avg Occupancy Rate by Sector & Season (%)')
ax.set_xlabel('')
ax.set_ylabel('')

plt.tight_layout()
plt.savefig('q2_temporal_trends.png', bbox_inches='tight')
#plt.show()
import os
os.makedirs("figures", exist_ok=True)

plt.tight_layout()
plt.savefig("figures/plot_199.png", dpi=300)  # 你也可以改成更有意义的名字
plt.close()
print("  Saved: q2_temporal_trends.png\n")


# =============================================================================
# Q3: HOW MUCH DOES UNAVAILABLE CAPACITY MAKE THINGS WORSE?
# =============================================================================
print("=" * 60)
print("Q3: Impact of Unavailable Capacity")
print("=" * 60)

# Compare official rate vs effective rate by sector
q3 = (
    df.groupby('SECTOR')
    .agg(
        Official_Rate   = ('OCCUPANCY_RATE',          'mean'),
        Effective_Rate  = ('EFFECTIVE_OCCUPANCY_RATE','mean'),
        Avg_Unavailable = ('UNAVAILABLE_CAPACITY',    'mean'),
    )
    .reset_index()
    .sort_values('Effective_Rate', ascending=False)
)
q3['Gap_pp'] = (q3['Effective_Rate'] - q3['Official_Rate']) * 100   # percentage point gap

print(q3.round(4).to_string(index=False))
print()

# Total beds lost to unavailability
total_unavailable = df['UNAVAILABLE_CAPACITY'].clip(lower=0).sum()
total_actual      = df['ACTUAL_CAPACITY'].sum()
pct_lost          = total_unavailable / total_actual * 100
print(f"  Total bed-days unavailable : {total_unavailable:,.0f}")
print(f"  Total bed-days available   : {total_actual:,.0f}")
print(f"  % of capacity lost         : {pct_lost:.2f}%")
print()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Q3: Impact of Unavailable Beds on Real System Pressure', fontsize=14, fontweight='bold', y=1.01)

# --- Chart A: Side-by-side official vs effective rate ---
ax = axes[0]
x = range(len(q3))
width = 0.35
bars1 = ax.bar([i - width/2 for i in x], q3['Official_Rate']  * 100,
               width, label='Official Rate',   color='steelblue', alpha=0.85)
bars2 = ax.bar([i + width/2 for i in x], q3['Effective_Rate'] * 100,
               width, label='Effective Rate\n(excl. unavailable)', color='tomato', alpha=0.85)
ax.axhline(100, color='red', linestyle='--', linewidth=1.2, label='100% capacity')
ax.set_xticks(list(x))
ax.set_xticklabels(q3['SECTOR'], rotation=15)
ax.set_ylabel('Occupancy Rate (%)')
ax.set_title('Official vs Effective Occupancy Rate by Sector')
ax.legend()
ax.set_ylim(85, 108)

# --- Chart B: Gap (percentage points) caused by unavailable beds ---
ax = axes[1]
bars = ax.bar(q3['SECTOR'], q3['Gap_pp'], color='tomato', alpha=0.85)
ax.set_xlabel('Sector')
ax.set_ylabel('Additional Pressure (percentage points)')
ax.set_title('Extra Pressure Added by Unavailable Beds\n(Effective Rate − Official Rate)')
plt.setp(ax.get_xticklabels(), rotation=15)
for bar, val in zip(bars, q3['Gap_pp']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'+{val:.2f}pp', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('q3_unavailable_capacity.png', bbox_inches='tight')
#plt.show()
plt.savefig("temp_plot.png"); plt.close()
print("  Saved: q3_unavailable_capacity.png\n")


# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("=" * 60)
print("SUMMARY: Key Findings")
print("=" * 60)

summary = sector_stats[['SECTOR','Avg_Occupancy_Rate_Pct','Pct_Days_Full_Pct']].copy()
summary.columns = ['Sector', 'Avg Occupancy Rate (%)', '% Days at Full Capacity']
summary = summary.merge(
    q3[['SECTOR','Effective_Rate','Gap_pp']].rename(columns={
        'SECTOR':'Sector',
        'Effective_Rate':'Effective Rate (incl. unavailable)',
        'Gap_pp':'Extra Pressure (pp)'
    }),
    on='Sector'
)
summary['Effective Rate (incl. unavailable)'] = (
    summary['Effective Rate (incl. unavailable)'] * 100
).round(2)
summary = summary.round(2)
print(summary.to_string(index=False))
print()
print("All charts saved as PNG files.")
print("Done!")
