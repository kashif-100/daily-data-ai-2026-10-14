# INTERVIEW_NOTES.md — Day 7: House Price Predictor

Honest Q&A prep for this exact project. Every answer describes what the code
actually does — no bluffing. Test results: 2,500 synthetic listings, test
R² = 0.983, RMSE = INR 4.0L, MAE = INR 3.2L.

## 30-second pitch

"I built a house price predictor in Python: it takes listing features — area,
bedrooms, zone, distance to the city centre and so on — and predicts the
price in lakhs using gradient boosting regression. On held-out data it
explains 98% of price variance with a typical error of about 4 lakhs. I also
built an explanation layer: permutation importance tells you exactly which
features drive each market segment's prices, and a narrative generator writes
that up as a plain-English valuation summary a buyer or agent could read."

## Project walkthrough (end to end)

**Q: Walk me through the project.**
First `generate_data.py` creates 2,500 synthetic listings from a fixed seed
(42), with a known pricing formula so the model can be sanity-checked against
ground truth. Then `house_price_predictor.py` one-hot encodes the
categoricals, splits 80/20, trains a GradientBoostingRegressor, evaluates on
the held-out set, ranks drivers with permutation importance, saves four
charts plus `findings.json`, and finally `ai_insights.py` converts the
metrics into a plain-English valuation report.

**Q: Why synthetic data instead of real listings?**
Two reasons: reproducibility — anyone can re-run and get the exact same
dataset from the seed — and a built-in answer key. Because I know the
ground-truth pricing formula, I can verify the model rediscovers the true
drivers (size, distance, bedrooms) instead of noise. The trade-off is that it
can't capture real market behaviour like neighbourhood reputation or
negotiation, so it demonstrates the method, not a production valuation.

**Q: What did the data look like?**
Ten features per listing: area (sqft), bedrooms, bathrooms, age, distance to
centre, floor, parking flag, balcony/garden flag, zone (5 categories) and
property type (4 categories). Prices ranged INR 10L–162.5L with a mean of
INR 70.4L. Bedrooms/bathrooms were derived from area with noise, and villas
are bigger by construction — so the features carry realistic correlations,
which matters when interpreting importance.

## Technical deep-dives

**Q: How does gradient boosting work, and why use it here?**
Gradient boosting builds trees sequentially, each new tree fitting the
*residuals* (mistakes) of the ensemble so far, with a small learning rate
shrinking each contribution to avoid overfitting. I chose it because the
pricing formula has genuine non-linearities — diminishing returns on area,
threshold premiums for villas and the Central zone — which a linear model
can't capture without hand-crafted features. Honest limitation: it's slower
to train than a linear model and its 300 trees are opaque, so I lean on
permutation importance for explainability.

**Q: How did you handle the categorical features?**
With a `ColumnTransformer` that applies `OneHotEncoder` to `zone` and
`property_type` and passes numerics through untouched, expanding 10 raw
features to 17 model inputs. The pipeline keeps preprocessing reproducible
and inside the training code rather than a fragile manual step. Honest
limitation: one-hot encoding drops the natural ordering (Central > North by
premium is learned, not encoded) and blows up dimensionality, but with only
9 categories it's harmless here.

**Q: What is permutation importance, and why not just use the model's built-in
feature importances?**
Permutation importance shuffles one feature at a time on the *held-out test
set* and measures how much R² drops — that drop is the honest price of
relying on that feature for generalisation. Built-in gradient-boosting
importances are computed on training data and can overstate features the
trees split on early but that don't generalise. Here permutation importance
ranked distance to centre (0.212 R² drop), area (0.197) and villa status (0.147)
at the top. Honest limitation: when features are correlated (area and
bedrooms are), shuffling one undercounts it because the other still carries
the signal — so the ranking is directional, not exact.

**Q: How did you evaluate the model?**
R² on the 20% held-out test split (0.983 — 98.3% of price variance
explained), RMSE (INR 4.0L, the typical error magnitude) and MAE
(INR 3.2L, the typical absolute error). I also report a baseline — always
predicting the training mean — which misses by INR 30.7L, so the model cuts
error by ~87%; a model is only impressive relative to a dumb alternative.
Honest limitation: R² this high partly reflects the clean synthetic formula
with mild noise; real listings would score lower.

**Q: How do you know the model isn't systematically biased?**
The residual plot (residual = actual − predicted, vs predicted price) scatters
evenly around zero with no funnel or curve, meaning errors don't grow or lean
in any price band. A funnel shape would signal heteroscedasticity — typical
errors growing with price — and a curve would signal a missed non-linearity.
Honest limitation: this checks *average* behaviour; it can't reveal
underperformance on a rare subgroup the data barely represents.

## Data-analyst thinking

**Q: What were the KPIs?**
Model accuracy KPIs: test R² (0.983), RMSE (INR 4.0L), MAE (INR 3.2L), and
error reduction vs baseline (~87%). Business-facing KPIs derived from the
model: the top-5 price drivers with their importance magnitudes, and the
zone-level price distributions (Central zone carries a ~INR 22L premium in the
ground truth). These turn a black-box number into defensible reasoning an
agent can quote to a client.

**Q: What was the single most important insight?**
Size and location dominate everything: area and distance to the city centre
together explain more of the price than all other features combined, while
age, parking and balcony are small but predictable add-ons. That tells a
buyer where negotiation leverage lives — distance and zone are fixed, so the
question is whether the *price per square foot* is fair for that zone, not
whether the balcony justifies a premium.

**Q: How did the analysis become recommendations?**
Permutation importance identifies *what* moves price, so the narrative
converts each driver into an action: buyers compare asking price to the
model's estimate and treat anything ~2× MAE (about INR 6.4L) above it as
overpriced; sellers justify premiums using the features the model values
most (size, zone) rather than emotional anchors. The recommendations are
explicitly framed as starting points with error bars, because a point
estimate used as a final offer is a misuse of the tool.

## The "AI integrated" part

**Q: What does "AI integrated" mean in this project?**
Two layers: the ML model itself (gradient boosting learning the pricing
function from data) and a deterministic narrative generator that converts
measured metrics into a plain-English valuation report with buyer/seller
takeaways. The generator is template-based and only fills in numbers from
`findings.json`, so every claim in the report is traceable to a measured
metric — it cannot hallucinate a figure the model never produced.

**Q: Why a template generator instead of an LLM for the narrative?**
Trade-offs. A template is deterministic, free, offline, and guaranteed to
only state measured numbers — ideal for a pipeline artifact. An LLM would
write more fluently and could answer follow-up questions, but it could invent
statistics and needs an API key and network. The honest answer is that the
current generator produces a *report*, not a *conversation*; upgrading the
narrative to an LLM summary over the same findings would be the natural next
step, keeping the numbers as grounded input.

**Q: What are the limitations of the whole approach?**
The data is synthetic, so nothing here transfers to a real market without
retraining. The model never sees amenities, schools, legal title or
construction quality — all real price movers. And errors are asymmetric in
real life: overpredicting costs a buyer money, underpredicting costs a
seller, so the point estimate needs human judgment and a margin, never blind
trust.

## With more time I would…

- Run a proper hyperparameter search (GridSearchCV over depth, learning rate
  and tree count) with cross-validation instead of fixed, reasonable
  hyperparameters, and report the tuning curve.
- Add SHAP values for per-listing explanations — "this flat costs INR 8L more
  than average *because* it's Central zone and 300 sqft larger" — which is
  more actionable than global importance for individual negotiations.
- Benchmark against linear regression and random forest on the same split to
  justify gradient boosting empirically rather than by argument.
- Simulate a market shock (e.g. a rate hike shifting the noise and zone
  premiums) to test how gracefully the model degrades, since models trained
  on stable data fail silently when the world changes.
