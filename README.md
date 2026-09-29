# Day-Ahead Electricity Price Forecasting (IEX)

Forecasts the clearing price of all **96 fifteen-minute blocks** of the next day
across three Indian Energy Exchange markets — DAM, GDAM and RTM — and flags, for
every block, which market is expected to be cheapest and which dearest.

The system runs unattended once a day, writes a colour-coded workbook for the
trading desk, and scores its own past predictions against the prices that
actually cleared.

## Results

Walk-forward backtest, out-of-sample, January–July 2026:

| Market | MAE (₹) | RMSE (₹) | WMAPE | Accuracy | R² | Cap capture |
|--------|--------:|---------:|------:|---------:|-----:|------------:|
| DAM    | 498     | 1,031    | 10.9% | 89.1%    | 0.890 | 90% |
| GDAM   | 392     | 778      | 7.9%  | 92.1%    | 0.936 | 92% |
| RTM    | 1,116   | 1,774    | 27.2% | 72.8%    | 0.606 | 60% |


Live Market Validation over 1 week- On the market-selection decision the system exists to support, it picks the
correct dearest market **79%** of the time and the correct cheapest **81%** of
the time, counting a pick as correct when it lands within 5% of the true best
price.

Two caveats worth stating plainly:

- **The headline is flattered by capped blocks.** A large share of summer blocks
  sit at the ₹10,000/MWh regulatory ceiling, and those are trivially easy once
  the classifier flags them. On non-capped blocks only, DAM runs around 84% and
  GDAM around 89%.
- **RTM is genuinely hard.** Its volatility comes from same-day events — a plant
  tripping, a demand surge — which leave no trace in the data available the day
  before. 73% is close to the honest ceiling for a day-ahead model, not a bug
  left unfixed.

## How it works

**Leakage-safe timing.** The three markets are not equally fresh at forecast
time. DAM and GDAM for the current day cleared the previous afternoon, so a
one-day lookback gives the model something real. RTM is still trading through
the day, so its most recent complete day is the day before yesterday — a two-day
lookback. This difference is encoded explicitly per market, so the model never
uses a number it would not genuinely have had.

**Two models per market.** A regressor predicts ordinary price movement. Left
alone it systematically under-shoots the price cap, because predicting the
ceiling is a large mistake on the many normal days. So a separate classifier
estimates the probability a block hits the cap, and a gate blends the final
prediction toward the ceiling weighted by that confidence. The two split the
work between the normal regime and the scarcity regime.

**Features.** 76 per market: price lags and rolling statistics, previous-day
summaries, bid and cleared volumes, the purchase-minus-sell gap as a scarcity
read, cap-regime persistence terms, calendar terms, weather from Open-Meteo, and
cross-market spreads. Each market's model sees the history of all three — DAM
and GDAM correlate at 0.90 and RTM tracks DAM at 0.81. The spreads turned out to
be the single most important feature for both GDAM and RTM: the relationship
between two markets is far more stable than the absolute level of either.

**Model.** Gradient-boosted trees (XGBoost), validated by walk-forward testing
that trains only on data preceding the period it scores. Each regressor predicts
the *change* from the most recent known price rather than the price itself.

## Layout

```
iex_dayahead/
  config.py     paths and token, all overridable by environment variable
  forecast.py   daily run: fetch -> features -> predict -> Excel -> backfill
  retrain.py    extend history, rebuild features, backtest, save models
  plots.py      forecast vs actual for a given day
```

## Running it

```bash
pip install -r requirements.txt
export IEX_TOKEN=your_token_here

python -m iex_dayahead.retrain              # build models (check the backtest table)
python -m iex_dayahead.forecast             # forecast tomorrow
python -m iex_dayahead.forecast --backfill-only   # fill in actuals, refresh accuracy
python -m iex_dayahead.plots 2026-07-24     # plot one day
```

## Figures

**Why the three markets are forecast together.** DAM and GDAM correlate at 0.90,
and even the far more erratic RTM tracks DAM at 0.81. Prices that move together
like this carry information about one another, which is why each market's model
is allowed to see the history of all three.

![Cross-market correlation](docs/figures/cross_market_correlation.png)

**The shape every model has to learn.** A trough in the middle of the day when
solar floods the grid, and a steep climb into the evening peak as the sun goes
and demand holds up.

![Intraday price shape](docs/figures/intraday_price_shape.png)

**Why a separate cap classifier is worth having.** Cap events are not scattered
randomly through the day - they cluster sharply in the evening peak, which is
what makes them predictable at all.

![Cap frequency by hour](docs/figures/cap_frequency_by_hour.png)

**Forecast against actual.** A cap-heavy day across all three markets: prices
sat at the ceiling for long stretches, crashed through the solar hours, and
climbed back. The forecast follows the shape closely, including lifting itself
onto the cap where the classifier fires.

![Forecast vs actual](docs/figures/forecast_vs_actual.png)

**The headline, and the honest version of it.** Overall accuracy against
accuracy on non-capped blocks only. The gap is the part of the headline that
comes from easy capped blocks.

![Overall vs non-cap accuracy](docs/figures/overall_vs_noncap_accuracy.png)

**Where the system is weakest.** The share of RTM cap events the model
anticipates, month by month. The instability is the point: RTM caps are
real-time surprises, and no day-ahead model can reliably foresee them.

![RTM cap detection by month](docs/figures/rtm_cap_detection.png)

**Consistency across the year.** DAM and GDAM hold a narrow band; RTM is lower
and more variable, as expected. March is the one visible dip - not caps, but an
unusually unsettled market, and a model anchored on the previous day struggles
when each day looks little like the last.

![Monthly accuracy](docs/figures/monthly_accuracy.png)

## Data

- **Prices**: Indian Energy Exchange public trade-data API (requires a token).
- **Weather**: [Open-Meteo](https://open-meteo.com/), averaged over
  representative solar, wind and load centres.

No data files are committed. `data/`, `models/` and `output/` are gitignored.

## What I would do next

The one thing the system cannot see is what is happening on the grid in real
time, and that is precisely what would most help RTM. Live generation and outage
information is the natural next input.

Two things that did *not* work, recorded because negative results shaped the
design as much as the successes did: adding grid demand and generation data gave
no lift, because price already carries that information; and a mirror of the cap
mechanism aimed at catching price crashes went nowhere, because crashes lack the
day-to-day persistence that makes caps predictable.

The most instructive bug was in deployment, not modelling. The live system
quietly reported accuracy about fifteen points below backtest — no error, no
warning, just worse predictions. A handful of features were not being rebuilt
correctly in the live path. Finding it meant comparing the two pipelines feature
by feature for a single day. I now treat that comparison as a routine deployment
check rather than a debugging step of last resort.
