# Assumptions, limitations and next steps

## Assumptions (all in `src/crm_bi/config.py`)
| Assumption | Value | Why it matters |
|---|---|---|
| Revenue per customer | €60 + 1.2% of balance per year | Sets how much a saved customer is worth |
| Horizon | 3 years | Longer horizons make retention more attractive |
| Contact cost | €45 | Drives where the value-gated cut-off falls |
| Save rate | 20% | Weakest assumption; break-even for a fixed-size campaign is 16.5% |
| Capacity | 20% of customers | Limits fixed-size campaigns |
| Cross-sell margin / baseline acceptance | €35 / 3% | Scale the NBO scenario |

## Method notes
- Hold-out is used once, after model choice on cross-validation.
- Calibration is chosen on out-of-fold Brier score (not applied here).
- The value-gated confidence interval comes from a bootstrap over hold-out customers; the difference versus random uses the same resamples.
- Complaint themes for the money view come from cross-validated supervised predictions, so no complaint is scored by a model that saw it.

## Limitations
Synthetic data, small hold-out, assumed economics, offline-only offer evaluation, and clustering settings chosen with access to true labels. See the README.

## Next steps
1. Randomised pilot for retention calls and for offers.
2. Uplift modelling to target customers who can be persuaded, not merely those likely to leave.
3. Survival analysis for time-to-churn and re-contact timing.
4. Sentence-embedding complaint clustering (needs a model download; not run by default).
5. Monitoring: population-stability checks and monthly retraining as in the architecture diagram.
