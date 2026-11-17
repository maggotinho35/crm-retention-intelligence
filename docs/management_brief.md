# Management brief: CRM retention programme

**Audience:** Head of CRM, Head of Retail Banking. **Basis:** hold-out results from `scripts/run_pipeline.py` (seed 42, synthetic data). Euro figures are scenarios.

## Recommendation
Launch a **value-gated retention pilot**, fix **billing and fee communication**, start **complaint tagging**, and test **model-based offers** against the current script, all with a randomised control group.

## Decision table

| # | Action | Measured evidence | Scenario impact | Main risk | Owner |
|---|---|---|---|---|---|
| 1 | Call only customers whose expected saved revenue exceeds the contact cost | AUC 0.83; 29 of 300 customers selected; 90% of them would have left | +€3.0k per 1,000 customers vs -€5.3k for random calls | Save rate below 16.5% makes fixed-size campaigns lose money | Retention team |
| 2 | Fix billing and fee issues | 29% of complaints, 34% of lost revenue; complainers churn 40% vs 25% | Not quantified here; billing complainers account for the largest share of lost revenue (see outputs/complaint_themes_value.csv) | Fee changes are also revenue; test messaging before price | CRM management |
| 3 | Tag 150+ complaints, then automate theme reporting | Supervised macro-F1 0.91 at 150 tags, 0.93 at 300+; unsupervised ARI 0.13 | Reliable root-cause reporting | Tag quality and label drift | Contact centre |
| 4 | Replace lift-ranked rules with a propensity model | hit@2 76.8% vs 70.0% script (CI +3.7 to +9.8 points); lift rules 54.2% | About +€100 per 1,000 contacts (CI €55 to €147) | Offline hit rate is not acceptance | Product / CRM |
| 5 | Build on the Microsoft stack with portable Python models | B 4.20 vs 3.50 and 3.35; wins 91% of simulations; flips at 41% cost weight | Vendor-risk limited by portable code | Cost becoming the top priority | IT / BI |

## Pilot design (recommended next step)
- Randomise the eligible gated list 50/50 into contact and control groups for 90 days.
- Primary metric: churn rate difference between groups. Secondary: revenue retained, cost per saved customer.
- Success threshold: save rate above 16.5% (break-even for a fixed-size campaign in the base case).
- Same design for offers: randomise script vs propensity model and compare acceptance.

## Assumptions behind the euro figures
See `src/crm_bi/config.py`: customer value = (€60 + 1.2% of average balance) x 3 years; contact cost €45; save rate 20%; capacity 20% of customers; cross-sell margin €35; baseline acceptance 3%. Sensitivity grid: `outputs/sensitivity_grid.csv`.
