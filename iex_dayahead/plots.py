"""Plot forecast vs actual for a single day from the forecast log.

Usage:  python -m iex_dayahead.plots 2026-07-24
        python -m iex_dayahead.plots            # defaults to yesterday
"""
import os
import sys
import datetime

import matplotlib
matplotlib.use("Agg")

from .config import OUT_DIR

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DAY = sys.argv[1] if len(sys.argv) > 1 else str(datetime.date.today() - datetime.timedelta(days=1))
OUT = str(OUT_DIR)

log = pd.read_csv(str(OUT_DIR / "dayahead_forecast_log.csv"), parse_dates=["target_date"])
dd = log[log["target_date"]==DAY].sort_values("block").reset_index(drop=True)
markets = ["dam","gdam","rtm"]
print("RTM actuals:", dd["rtm_act"].notna().sum(), "/96")

# ---- PNG ----
fig, axes = plt.subplots(3,1, figsize=(11,9), sharex=True)
for ax, m in zip(axes, markets):
    ax.plot(dd["block"], dd[f"{m}_fc"],  "--", color="#c00000", label="Forecast", lw=2)
    ax.plot(dd["block"], dd[f"{m}_act"], color="#1f3864", label="Actual", lw=2)
    ax.set_title(f"{m.upper()} — {DAY}"); ax.set_ylabel("Rs/MWh"); ax.legend(); ax.grid(alpha=0.3)
axes[-1].set_xlabel("Block (0-95)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, f"plot_{DAY.replace('-','')}.png"), dpi=120)
plt.close()

# ---- interactive Plotly HTML ----
times = dd["time"].tolist()
figp = make_subplots(rows=3, cols=1, shared_xaxes=True,
                     subplot_titles=[m.upper() for m in markets], vertical_spacing=0.07)
for r, m in enumerate(markets, start=1):
    figp.add_trace(go.Scatter(x=dd["block"], y=dd[f"{m}_fc"], name=f"{m.upper()} forecast",
        mode="lines", line=dict(color="#c00000", dash="dash", width=2), customdata=times,
        hovertemplate="Block %{x} (%{customdata})<br>Forecast: ₹%{y:,.0f}<extra></extra>"), row=r, col=1)
    figp.add_trace(go.Scatter(x=dd["block"], y=dd[f"{m}_act"], name=f"{m.upper()} actual",
        mode="lines", line=dict(color="#1f3864", width=2), customdata=times,
        hovertemplate="Block %{x} (%{customdata})<br>Actual: ₹%{y:,.0f}<extra></extra>"), row=r, col=1)
    figp.update_yaxes(title_text="Rs/MWh", row=r, col=1)
figp.update_xaxes(title_text="Block (0-95)", row=3, col=1)
figp.update_layout(height=850, width=1000, hovermode="x unified",
                   title=f"Day-Ahead Forecast vs Actual — {DAY}",
                   template="plotly_white", legend=dict(orientation="h", y=-0.08))
figp.write_html(os.path.join(OUT, f"plot_{DAY.replace('-','')}.html"))

print("saved PNG and HTML for", DAY)
