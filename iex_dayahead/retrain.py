
# # PRICE_DB — Retrain & Refresh Models (clean pipeline)
# 
# **What this notebook does, top to bottom, in one run:**
# 1. Reuses the live pipeline's IEX fetcher, so training and serving agree
# 2. Backs up `iex_history.csv`, then extends it with any new days from the IEX API
# 3. Builds all features (price, calendar, volume, spread, cap-regime, weather)
# 4. Runs the walk-forward backtest (**check accuracy here before saving**)
# 5. Saves the 6 models + config to `dayahead_models\`
# 
# **To use:** run top to bottom. Then inspect the backtest table in the
# "Backtest" section; if accuracy is sound, the save cell has already written the
# models. If accuracy looks wrong, do **not** deploy — investigate first.
# 
# 

# Step 0 - shared data fetchers
from .config import DATA_DIR, MODEL_DIR, OUT_DIR, HISTORY_CSV, WEATHER_CSV
from .forecast import fetch_iex


# ## Step 1 — Back up and extend `iex_history.csv` from the API
# Fetches every day from the CSV's last date up to yesterday, appends, de-duplicates, saves. If the CSV is already current, it simply refetches the last day and changes nothing.

import pandas as pd, numpy as np, datetime, shutil

CSV = str(HISTORY_CSV)

# back up first (timestamped, so repeated runs don't overwrite the same backup)
_bk = CSV.replace(".csv", f"_backup_{datetime.datetime.now():%Y%m%d_%H%M}.csv")
shutil.copy(CSV, _bk); print("backup:", _bk)

hist = pd.read_csv(CSV); hist["Date"] = pd.to_datetime(hist["Date"])
last  = hist["Date"].max()
today = pd.Timestamp(datetime.date.today())
frm = last.strftime('%d/%m/%Y')                      # refetch last day (it may be partial)
to  = (today - pd.Timedelta(days=1)).strftime('%d/%m/%Y')

if last.date() >= (today - pd.Timedelta(days=1)).date():
    print(f"CSV already current (ends {last.date()}). Nothing to fetch.")
else:
    print(f"CSV ends {last.date()} | fetching {frm} -> {to}")
    frames = {}
    for m, pref in [("dam","DAM"), ("gdam","GDAM"), ("rtm","RTM")]:
        a = fetch_iex(m, frm, to)
        a = a.rename(columns={"mcp": f"{pref}_MCP", "pb": f"{pref}_PurchaseBid_MW",
                              "sb": f"{pref}_SellBid_MW", "mcv": f"{pref}_MCV_MW"})
        frames[m] = a[["date","block",f"{pref}_MCP",f"{pref}_PurchaseBid_MW",
                       f"{pref}_SellBid_MW",f"{pref}_MCV_MW"]]
    new = frames["dam"].merge(frames["gdam"], on=["date","block"], how="outer") \
                       .merge(frames["rtm"],  on=["date","block"], how="outer")
    new["Date"]  = pd.to_datetime(new["date"])
    new["Block"] = new["block"].astype(int) + 1
    mins = new["block"]*15
    new["TimeStart"] = mins.apply(lambda x: f"{x//60:02d}:{x%60:02d}")
    new["TimeEnd"]   = ((mins+15)%1440).apply(lambda x: f"{x//60:02d}:{x%60:02d}")
    new = new[hist.columns]
    hist = hist[hist["Date"] < last]                 # drop the old partial last day
    combined = pd.concat([hist, new], ignore_index=True) \
                 .drop_duplicates(subset=["Date","Block"], keep="last") \
                 .sort_values(["Date","Block"]).reset_index(drop=True)
    assert list(combined.columns) == list(hist.columns), "column mismatch!"
    combined.to_csv(CSV, index=False)
    print(f"SAVED. now ends {combined['Date'].max().date()} | {len(combined):,} rows")


# ## Step 2 — Build features

# ============================================================
# CELL 1 — load iex_history.csv and tidy the three markets
# ============================================================
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

IEX_FILE = str(HISTORY_CSV)

df = pd.read_csv(IEX_FILE, parse_dates=["Date"])
df["date"]  = df["Date"].dt.normalize()
df["block"] = df["Block"].astype(int) - 1        # 1..96 -> 0..95

cols = {
    "DAM_MCP":"dam", "GDAM_MCP":"gdam", "RTM_MCP":"rtm",
    "DAM_PurchaseBid_MW":"dam_pb", "DAM_SellBid_MW":"dam_sb", "DAM_MCV_MW":"dam_mcv",
    "GDAM_PurchaseBid_MW":"gdam_pb", "GDAM_SellBid_MW":"gdam_sb", "GDAM_MCV_MW":"gdam_mcv",
    "RTM_PurchaseBid_MW":"rtm_pb", "RTM_SellBid_MW":"rtm_sb", "RTM_MCV_MW":"rtm_mcv",
}
d = df[["date","block"]+list(cols.keys())].rename(columns=cols).sort_values(["date","block"]).reset_index(drop=True)

print("rows:", len(d), " days:", d['date'].nunique(), " blocks/day:", d.groupby('date').size().unique())
print("date range:", d['date'].min().date(), "to", d['date'].max().date())
print("\nmissing values per market:")
print(d[["dam","gdam","rtm"]].isna().sum())


# ============================================================
# CELL 2 — day-ahead price features with market-correct timing
#   DAM/GDAM: cleared the day before -> shift(1) is available at 9am
#   RTM:      real-time -> only shift(2)+ is fully available at 9am
# ============================================================
markets = ["dam", "gdam", "rtm"]
grids = {m: d.pivot(index="date", columns="block", values=m).sort_index() for m in markets}
for m in markets:
    assert grids[m].shape[1] == 96, f"{m}: expected 96 blocks"

BASE_SHIFT = {"dam": 1, "gdam": 1, "rtm": 2}

def build_price_features():
    feats = {}
    for m in markets:
        P = grids[m]
        s = BASE_SHIFT[m]
        feats[f"{m}_lag_recent"] = P.shift(s)
        feats[f"{m}_lag_prev"]   = P.shift(s + 1)
        feats[f"{m}_lag7d"]      = P.shift(7)
        feats[f"{m}_lag14d"]     = P.shift(14)
        feats[f"{m}_roll7_mean"]  = P.shift(s).rolling(7).mean()
        feats[f"{m}_roll7_std"]   = P.shift(s).rolling(7).std()
        feats[f"{m}_roll14_mean"] = P.shift(s).rolling(14).mean()
    return feats

price_feats = build_price_features()
print("built", len(price_feats), "price-history feature grids")
print("shifts used:", BASE_SHIFT)

tb, pb = grids["rtm"].index[20], grids["rtm"].index[18]
lhs = price_feats["rtm_lag_recent"].loc[tb, 40]
rhs = grids["rtm"].loc[pb, 40]
print(f"\nRTM leakage check (block 40): lag_recent on {tb.date()} = {lhs:.1f}, "
      f"actual 2 days earlier ({pb.date()}) = {rhs:.1f} -> {'OK' if abs(lhs-rhs)<0.01 else 'MISMATCH'}")


# ============================================================
# CELL 3 — calendar + previous-available-day summaries
# ============================================================
import holidays as hl

long = d[["date","block","dam","gdam","rtm"]].copy().sort_values(["date","block"]).reset_index(drop=True)

long["hour"]    = long["block"] // 4
long["dow"]     = long["date"].dt.dayofweek
long["is_wknd"] = (long["dow"] >= 5).astype(int)
long["month"]   = long["date"].dt.month
long["doy"]     = long["date"].dt.dayofyear
long["sin_blk"] = np.sin(2*np.pi*long["block"]/96)
long["cos_blk"] = np.cos(2*np.pi*long["block"]/96)
long["sin_doy"] = np.sin(2*np.pi*long["doy"]/365)
long["cos_doy"] = np.cos(2*np.pi*long["doy"]/365)
in_hol = pd.to_datetime(list(hl.India(years=range(2023,2028)).keys()))
long["is_hol"]  = long["date"].isin(in_hol).astype(int)

for m in markets:
    P = grids[m]
    s = BASE_SHIFT[m]
    dl = pd.DataFrame({
        f"{m}_daymean": P.shift(s).mean(axis=1),
        f"{m}_daymax":  P.shift(s).max(axis=1),
        f"{m}_daymin":  P.shift(s).min(axis=1),
        f"{m}_evening": P.shift(s)[list(range(68,88))].mean(axis=1),
        f"{m}_midday":  P.shift(s)[list(range(44,56))].mean(axis=1),
    })
    long = long.merge(dl.reset_index(), on="date", how="left")

print("long shape:", long.shape)


# ============================================================
# CELL 3B — bid-volume + cross-market spread features (test)
# ============================================================
vol_cols = ["dam_pb","dam_sb","dam_mcv","gdam_pb","gdam_sb","gdam_mcv","rtm_pb","rtm_sb","rtm_mcv"]
vgrids = {c: d.pivot(index="date", columns="block", values=c).sort_index() for c in vol_cols}

vol_feats = {}
for c in vol_cols:
    m = c.split("_")[0]                    
    s = BASE_SHIFT[m]                     
    vol_feats[f"{c}_lag"] = vgrids[c].shift(s)

for m in markets:
    s = BASE_SHIFT[m]
    vol_feats[f"{m}_sdgap_lag"] = vgrids[f"{m}_pb"].shift(s) - vgrids[f"{m}_sb"].shift(s)

for name, grid in vol_feats.items():
    melted = grid.reset_index().melt(id_vars="date", var_name="block", value_name=name)
    long = long.merge(melted, on=["date","block"], how="left")

sp1 = price_feats["dam_lag_recent"] - price_feats["gdam_lag_recent"]
sp2 = price_feats["dam_lag_recent"] - price_feats["rtm_lag_recent"]
for name, grid in [("spread_dam_gdam", sp1), ("spread_dam_rtm", sp2)]:
    melted = grid.reset_index().melt(id_vars="date", var_name="block", value_name=name)
    long = long.merge(melted, on=["date","block"], how="left")

new_feats = list(vol_feats.keys()) + ["spread_dam_gdam","spread_dam_rtm"]
print("added", len(new_feats), "new features:", new_feats)


# ============================================================
# CELL 3C — cap-regime persistence features (leakage-safe)
# ============================================================
CAP_LVL = 9900

for m in markets:
    P = grids[m]; s = BASE_SHIFT[m]
    capped = (P >= CAP_LVL).astype(float)

    day_cap_count = capped.sum(axis=1)                       
    eve_capped    = (capped[list(range(68,88))].sum(axis=1) > 0).astype(int)
    streak = (day_cap_count > 0).astype(int)
    streak = streak.groupby((streak == 0).cumsum()).cumsum() 
    dl = pd.DataFrame({
        f"{m}_capcount_recent": day_cap_count.shift(s),
        f"{m}_evecap_recent":   eve_capped.shift(s),
        f"{m}_capstreak":       streak.shift(s),
    })
    long = long.merge(dl.reset_index(), on="date", how="left")

    blk_freq = capped.shift(s).rolling(7).mean()
    melted = blk_freq.reset_index().melt(id_vars="date", var_name="block",
                                         value_name=f"{m}_blkcapfreq7")
    long = long.merge(melted, on=["date","block"], how="left")

new_cap_feats = [f"{m}_{x}" for m in markets
                 for x in ["capcount_recent","evecap_recent","capstreak","blkcapfreq7"]]
print("added", len(new_cap_feats), "cap-regime features")
print(long[long["date"]=="2026-06-15"][["block"]+new_cap_feats[:4]].head(3).to_string(index=False))


# ============================================================
# CELL 4 — merge price-history + weather; define feature lists for ALL markets
# ============================================================
import os

for name, grid in price_feats.items():
    melted = grid.reset_index().melt(id_vars="date", var_name="block", value_name=name)
    long = long.merge(melted, on=["date","block"], how="left")
print("after price-history merge:", long.shape)

WX_FILE = str(WEATHER_CSV)
if os.path.exists(WX_FILE):
    wx = pd.read_csv(WX_FILE)
    wx["date"] = pd.to_datetime(wx["date"]).dt.normalize()
    wx["hour"] = wx["hour"].astype(int)
    long = long.merge(wx[["date","hour","ghi","temp","cloud","wind100"]], on=["date","hour"], how="left")
    print("weather merged. rows with weather:", long["ghi"].notna().sum(), "/", len(long))
else:
    print("weather file not found -- proceeding without weather")
    for c in ["ghi","temp","cloud","wind100"]:
        long[c] = np.nan

calendar_feats = ["block","hour","dow","is_wknd","month","sin_blk","cos_blk","sin_doy","cos_doy","is_hol"]
weather_feats  = ["ghi","temp","cloud","wind100"]

def market_features(target_m):
    hist = []
    for m in markets:
        hist += [c for c in long.columns if c.startswith(f"{m}_") and c != m]
    return hist + calendar_feats + weather_feats

dam_features  = market_features("dam")
gdam_features = market_features("gdam")
rtm_features  = market_features("rtm")
dam_features  += new_feats
gdam_features += new_feats
rtm_features  += new_feats
market_setup = {
    "dam":  ("dam_lag_recent",  dam_features),
    "gdam": ("gdam_lag_recent", gdam_features),
    "rtm":  ("rtm_lag_recent",  rtm_features),
}
print("feature counts:", {m: len(f) for m,(a,f) in market_setup.items()})


# ## Step 3 — Params, dedupe, and backtest helpers

# ============================================================
# CELL 6 — BEST PARAMS (always runs)
# Loads best_params.json if the tuning cell has ever been run;
# otherwise falls back to the validated baseline defaults.
# ============================================================
import json, os
import xgboost as xgb

PARAMS_FILE = str(MODEL_DIR / "best_params.json")

DEFAULT_PARAMS = {
    "dam":  dict(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.7,
                 colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0),
    "gdam": dict(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.7,
                 colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0),
    "rtm":  dict(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.7,
                 colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0),
}

if os.path.exists(PARAMS_FILE):
    with open(PARAMS_FILE) as f:
        BEST_PARAMS = json.load(f)
    src = f"loaded from {PARAMS_FILE}"
else:
    BEST_PARAMS = DEFAULT_PARAMS
    src = "defaults (tuning cell has not been run yet)"

def make_reg_m(m):
    return xgb.XGBRegressor(**BEST_PARAMS[m], tree_method="hist", n_jobs=-1, random_state=42)

print("params source:", src)
for m, pr in BEST_PARAMS.items():
    print(f"  {m.upper()}: {pr}")


def dedupe(lst):
    seen=set(); out=[]
    for c in lst:
        if c not in seen: seen.add(c); out.append(c)
    return out

dam_features  = dedupe(dam_features)
gdam_features = dedupe(gdam_features)
rtm_features  = dedupe(rtm_features)

market_setup = {
    "dam":  ("dam_lag_recent",  dam_features),
    "gdam": ("gdam_lag_recent", gdam_features),
    "rtm":  ("rtm_lag_recent",  rtm_features),
}
print("counts:", {m: len(f) for m,(a,f) in market_setup.items()})
print("cap feats:", sum(1 for c in new_cap_feats if c in dam_features), "/ 12")
print("duplicates:", any(len(f) != len(set(f)) for _,(a,f) in market_setup.items()))


import numpy as np, pandas as pd, xgboost as xgb

TRAIN_END = pd.Timestamp("2025-12-31")
CAP_THRESHOLD = 9900
GATE = 0.5
CAP_VALUE = 10000.0

def metrics(yt, yp):
    yt, yp = np.asarray(yt,float), np.asarray(yp,float)
    e = yp - yt
    return np.mean(np.abs(e)), np.sqrt(np.mean(e**2)), 100*np.sum(np.abs(e))/np.sum(np.abs(yt))

def make_reg_m(m):
    return xgb.XGBRegressor(**BEST_PARAMS[m], tree_method="hist", n_jobs=-1, random_state=42)

def make_clf():
    return xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                             reg_lambda=2.0, tree_method="hist", n_jobs=-1,
                             random_state=42, eval_metric="logloss")

# verify everything 9B needs is now present
need = ["metrics","make_reg_m","make_clf","BEST_PARAMS","market_setup","long",
        "TRAIN_END","CAP_THRESHOLD","GATE","CAP_VALUE"]
for n in need:
    print(f"{n:16s}", "OK" if n in dir() else "*** MISSING ***")


# ## Step 4 — Backtest  ⚠️  CHECK THIS TABLE BEFORE SAVING
# Expected baseline (pre-July): DAM ~89 · GDAM ~92 · RTM ~74. With July now included the numbers may shift. If accuracy is materially worse, investigate before deploying.

# ============================================================
# CELL 9B — walk-forward with cap classifier + gate for ALL markets
# ============================================================
def walk_forward_gated(df, feats, target, anchor, train_end, retrain_every=30):
    df = df.dropna(subset=[target, anchor]).sort_values(["date","block"]).reset_index(drop=True)
    df = df.assign(is_cap=(df[target] >= CAP_THRESHOLD).astype(int))
    days = sorted(df.loc[df["date"] > train_end, "date"].unique())
    out, reg, clf, since = [], None, None, 10**9
    for D in days:
        if since >= retrain_every:
            tr = df[df["date"] < D]
            reg = make_reg_m(target)
            reg.fit(tr[feats].values, (tr[target] - tr[anchor]).values)
            clf = make_clf()
            clf.fit(tr[feats].values, tr["is_cap"].values)
            since = 0
        day = df[df["date"] == D]
        base  = day[anchor].values + reg.predict(day[feats].values)
        cap_p = clf.predict_proba(day[feats].values)[:,1]
        yhat  = np.where(cap_p >= GATE, (1-cap_p)*base + cap_p*CAP_VALUE, base)
        recent = df.loc[(df["date"]<D)&(df["date"]>=D-pd.Timedelta(days=90)), target]
        hi = float(recent.max()) if len(recent) else 12000.0
        yhat = np.clip(yhat, 0, hi)
        out.append(pd.DataFrame({"date":day["date"].values,"block":day["block"].values,
                                 "y":day[target].values,"yhat":yhat,"cap_prob":cap_p}))
        since += 1
    return pd.concat(out, ignore_index=True)

baseline = {"dam": 88.5, "gdam": 91.8, "rtm": 73.8}
print(f"{'Market':>6} {'Baseline':>9} {'Gated+capfeats':>15} {'CapHit':>8}")
gated_preds = {}
for m, (anchor, feats) in market_setup.items():
    pred = walk_forward_gated(long, feats, m, anchor, TRAIN_END)
    gated_preds[m] = pred
    _,_,w = metrics(pred["y"], pred["yhat"])
    tc = pred["y"] >= CAP_THRESHOLD
    hit = (pred.loc[tc,"yhat"] >= 8000).mean() if tc.sum() else float("nan")
    print(f"{m.upper():>6} {baseline[m]:>8.1f}% {100-w:>14.1f}% {hit:>7.1%}  ({int(tc.sum())} caps)")


# ## Step 5 — Save models + config
# Writes the 6 model files and `dayahead_config.json`. The live script picks these up automatically on its next run.

# ============================================================
# SAVE — all 3 regressors + all 3 cap classifiers + config
# ============================================================
import os, json
SAVE_DIR = str(MODEL_DIR)
os.makedirs(SAVE_DIR, exist_ok=True)

FEATURES = {"dam": dam_features, "gdam": gdam_features, "rtm": rtm_features}
ANCHORS  = {"dam": "dam_lag_recent", "gdam": "gdam_lag_recent", "rtm": "rtm_lag_recent"}

for m in markets:
    full = long.dropna(subset=[m, ANCHORS[m]]).copy()
    reg = make_reg_m(m)
    reg.fit(full[FEATURES[m]].values, (full[m] - full[ANCHORS[m]]).values)
    reg.save_model(os.path.join(SAVE_DIR, f"{m}_regressor.json"))
    full["is_cap"] = (full[m] >= CAP_THRESHOLD).astype(int)
    clf = make_clf()
    clf.fit(full[FEATURES[m]].values, full["is_cap"].values)
    clf.save_model(os.path.join(SAVE_DIR, f"{m}_classifier.json"))
    print(f"{m.upper()}: regressor + classifier saved ({len(full):,} rows, {int(full['is_cap'].sum())} caps)")

config = {
    "markets": markets, "features": FEATURES, "anchors": ANCHORS,
    "base_shift": BASE_SHIFT, "calendar_feats": calendar_feats, "weather_feats": weather_feats,
    "cap_threshold": CAP_THRESHOLD, "gate": GATE, "cap_value": CAP_VALUE,
    "cap_feats": new_cap_feats, "best_params": BEST_PARAMS,
    "trained_through": str(long["date"].max().date()),
}
with open(os.path.join(SAVE_DIR, "dayahead_config.json"), "w") as f:
    json.dump(config, f, indent=2)
print("\nconfig saved with", len(new_cap_feats), "cap features recorded")


print(f"{'Market':>6} {'MAE':>7} {'WMAPE':>7} {'Acc':>7} {'R2':>7} {'CapHit':>7}")
for m in ["dam","gdam","rtm"]:
    g = gated_preds[m]
    e = (g["yhat"]-g["y"]).abs()
    w = 100*e.sum()/g["y"].abs().sum()
    ss_res=((g["yhat"]-g["y"])**2).sum(); ss_tot=((g["y"]-g["y"].mean())**2).sum()
    tc = g["y"]>=9900
    ch = (g.loc[tc,"yhat"]>=8000).mean()*100 if tc.sum() else float("nan")
    print(f"{m.upper():>6} {e.mean():>7,.0f} {w:>6.1f}% {100-w:>6.1f}% {1-ss_res/ss_tot:>7.3f} {ch:>6.0f}%")


for m in ["dam","gdam","rtm"]:
    g = gated_preds[m].copy(); g["mo"]=pd.to_datetime(g["date"]).dt.month
    print(f"\n=== {m.upper()} ===")
    print(f"{'month':>6} {'acc':>7} {'non-cap':>8} {'caps':>6} {'cap-hit':>8}")
    for mo in sorted(g["mo"].unique()):
        d=g[g["mo"]==mo]; non=d[d["y"]<9900]; tc=d["y"]>=9900
        a=100-100*(d["yhat"]-d["y"]).abs().sum()/d["y"].abs().sum()
        na=100-100*(non["yhat"]-non["y"]).abs().sum()/non["y"].abs().sum() if len(non) else float("nan")
        ch=(d.loc[tc,"yhat"]>=8000).mean()*100 if tc.sum() else float("nan")
        print(f"{mo:>6} {a:>6.1f}% {na:>7.1f}% {int(tc.sum()):>6} {ch:>7.0f}%")


for m in ["dam","gdam","rtm"]:
    reg=make_reg_m(m); full=long.dropna(subset=[m,f"{m}_lag_recent"])
    reg.fit(full[market_setup[m][1]].values,(full[m]-full[f"{m}_lag_recent"]).values)
    imp=pd.Series(reg.feature_importances_,index=market_setup[m][1]).sort_values(ascending=False)
    print(f"\n=== {m.upper()} top 12 ===")
    for f,v in imp.head(12).items(): print(f"  {f:<24} {v:.4f}")


import matplotlib.pyplot as plt
OUT=str(OUT_DIR)
COL={"dam":"#1f3864","gdam":"#2e8b57","rtm":"#c00000"}

# (a) monthly accuracy, all markets
fig,ax=plt.subplots(figsize=(10,5))
for m in ["dam","gdam","rtm"]:
    g=gated_preds[m].copy(); g["mo"]=pd.to_datetime(g["date"]).dt.month
    xs,ys=[],[]
    for mo in sorted(g["mo"].unique()):
        d=g[g["mo"]==mo]; xs.append(mo); ys.append(100-100*(d["yhat"]-d["y"]).abs().sum()/d["y"].abs().sum())
    ax.plot(xs,ys,marker="o",lw=2,color=COL[m],label=m.upper())
ax.set_xlabel("Month (2026)");ax.set_ylabel("Accuracy %");ax.set_title("Monthly forecast accuracy by market")
ax.legend();ax.grid(alpha=0.3);plt.tight_layout();plt.savefig(rf"{OUT}\rep_monthly_acc.png",dpi=120);plt.show()

# (b) intraday average shape
d=pd.read_csv(str(HISTORY_CSV))
d["b"]=d["Block"]-1
prof=d.groupby("b")[["DAM_MCP","GDAM_MCP","RTM_MCP"]].mean()
fig,ax=plt.subplots(figsize=(10,5))
for m,c in [("DAM_MCP","#1f3864"),("GDAM_MCP","#2e8b57"),("RTM_MCP","#c00000")]:
    ax.plot(prof.index,prof[m],lw=2,color=c,label=m.split("_")[0])
ax.set_xlabel("Block (0-95)");ax.set_ylabel("Avg MCP (Rs/MWh)");ax.set_title("Average intraday price shape")
ax.legend();ax.grid(alpha=0.3);plt.tight_layout();plt.savefig(rf"{OUT}\rep_intraday.png",dpi=120);plt.show()

# (c) cap frequency by hour
d["hr"]=d["b"]//4
cf=pd.DataFrame({m:(d.groupby("hr")[f"{m}_MCP"].apply(lambda x:(x>=9900).mean()*100)) for m in ["DAM","GDAM","RTM"]})
fig,ax=plt.subplots(figsize=(10,5))
for m,c in [("DAM","#1f3864"),("GDAM","#2e8b57"),("RTM","#c00000")]:
    ax.plot(cf.index,cf[m],marker="o",ms=3,lw=2,color=c,label=m)
ax.set_xlabel("Hour");ax.set_ylabel("% blocks at cap");ax.set_title("Cap-event frequency by hour")
ax.legend();ax.grid(alpha=0.3);plt.tight_layout();plt.savefig(rf"{OUT}\rep_capfreq.png",dpi=120);plt.show()

# (d) correlation heatmap
import numpy as np
corr=d[["DAM_MCP","GDAM_MCP","RTM_MCP"]].corr()
fig,ax=plt.subplots(figsize=(5,4))
im=ax.imshow(corr,cmap="Blues",vmin=0,vmax=1)
for i in range(3):
    for j in range(3): ax.text(j,i,f"{corr.iloc[i,j]:.2f}",ha="center",va="center",color="white" if corr.iloc[i,j]>0.6 else "black")
ax.set_xticks(range(3));ax.set_xticklabels(["DAM","GDAM","RTM"]);ax.set_yticks(range(3));ax.set_yticklabels(["DAM","GDAM","RTM"])
ax.set_title("Cross-market correlation");plt.colorbar(im);plt.tight_layout();plt.savefig(rf"{OUT}\rep_corr.png",dpi=120);plt.show()

print("saved 4 report graphs")


import matplotlib.pyplot as plt, numpy as np
OUT=str(OUT_DIR)
mk=["DAM","GDAM","RTM"]; head=[89.1,92.1,72.8]; noncap=[]
for m in ["dam","gdam","rtm"]:
    g=gated_preds[m]; non=g[g["y"]<9900]
    noncap.append(100-100*(non["yhat"]-non["y"]).abs().sum()/non["y"].abs().sum())
x=np.arange(3); w=0.36
fig,ax=plt.subplots(figsize=(8,5))
ax.bar(x-w/2,head,w,label="Overall",color="#1f3864")
ax.bar(x+w/2,noncap,w,label="Non-cap blocks only",color="#8fa9d0")
for i,(h,n) in enumerate(zip(head,noncap)):
    ax.text(i-w/2,h+0.6,f"{h:.1f}",ha="center",fontsize=9,fontweight="bold")
    ax.text(i+w/2,n+0.6,f"{n:.1f}",ha="center",fontsize=9,fontweight="bold")
ax.set_xticks(x);ax.set_xticklabels(mk);ax.set_ylabel("Accuracy %");ax.set_ylim(0,100)
ax.set_title("Overall vs non-cap accuracy");ax.legend();ax.grid(alpha=0.3,axis="y")
plt.tight_layout();plt.savefig(rf"{OUT}\rep_noncap.png",dpi=120);plt.show()


import pandas as pd
g=gated_preds["rtm"].copy(); g["mo"]=pd.to_datetime(g["date"]).dt.month
mos,hits=[],[]
for mo in sorted(g["mo"].unique()):
    d=g[g["mo"]==mo]; tc=d["y"]>=9900
    if tc.sum(): mos.append(mo); hits.append((d.loc[tc,"yhat"]>=8000).mean()*100)
fig,ax=plt.subplots(figsize=(9,4.5))
ax.bar(mos,hits,color="#c00000",alpha=0.75)
for x,h in zip(mos,hits): ax.text(x,h+1.5,f"{h:.0f}%",ha="center",fontsize=9)
ax.set_xlabel("Month (2026)");ax.set_ylabel("RTM cap-hit %");ax.set_ylim(0,100)
ax.set_title("RTM cap-detection varies sharply month to month")
ax.grid(alpha=0.3,axis="y");plt.tight_layout();plt.savefig(rf"{OUT}\rep_rtm_caphit.png",dpi=120);plt.show()


import numpy as np
print(f"{'Market':>6} {'RMSE':>8}")
for m in ["dam","gdam","rtm"]:
    g = gated_preds[m]
    rmse = np.sqrt(((g["yhat"]-g["y"])**2).mean())
    print(f"{m.upper():>6} {rmse:>8,.0f}")


import pandas as pd, numpy as np
log = pd.read_csv(str(OUT_DIR / "dayahead_forecast_log.csv"),
                  parse_dates=["target_date"])
print(f"{'Market':>6} {'blocks':>7} {'MAE':>8} {'RMSE':>8} {'WMAPE':>8} {'Acc':>7} {'R2':>7}")
for m in ["dam","gdam","rtm"]:
    d = log.dropna(subset=[f"{m}_act", f"{m}_fc"])
    if not len(d):
        print(f"{m.upper():>6}  no live data yet"); continue
    e = (d[f"{m}_fc"]-d[f"{m}_act"]).abs()
    rmse = np.sqrt(((d[f"{m}_fc"]-d[f"{m}_act"])**2).mean())
    w = 100*e.sum()/d[f"{m}_act"].abs().sum()
    ss_res=((d[f"{m}_fc"]-d[f"{m}_act"])**2).sum(); ss_tot=((d[f"{m}_act"]-d[f"{m}_act"].mean())**2).sum()
    r2 = 1-ss_res/ss_tot if ss_tot else float("nan")
    print(f"{m.upper():>6} {len(d):>7,} {e.mean():>8,.0f} {rmse:>8,.0f} {w:>7.1f}% {100-w:>6.1f}% {r2:>7.3f}")
print(f"\nlive date range: {log.dropna(subset=['dam_act'])['target_date'].min()} -> {log.dropna(subset=['dam_act'])['target_date'].max()}")
print(f"live days scored: {log.dropna(subset=['dam_act'])['target_date'].nunique()}")


import pandas as pd
log = pd.read_csv(str(OUT_DIR / "dayahead_forecast_log.csv"),
                  parse_dates=["target_date"])
scored = log.dropna(subset=["max_hit","max_hit_strict"])
print(f"scored blocks: {len(scored)}")
print(f"dearest-market pick — within 5%: {(scored['max_hit']=='Y').mean()*100:.0f}%  strict: {(scored['max_hit_strict']=='Y').mean()*100:.0f}%")
sc2 = log.dropna(subset=["min_hit","min_hit_strict"])
print(f"cheapest-market pick — within 5%: {(sc2['min_hit']=='Y').mean()*100:.0f}%  strict: {(sc2['min_hit_strict']=='Y').mean()*100:.0f}%")


