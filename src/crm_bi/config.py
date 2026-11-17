"""Central configuration: seeds, paths and the business ASSUMPTIONS used to translate model output into euros.

Everything in `BusinessAssumptions` is an illustrative planning assumption for the fictional
St. Maggotinho's Bank. None of it is measured. Change the numbers, re-run the pipeline, and every
euro figure in outputs/ and images/ updates. The sensitivity analysis shows how much the
conclusions depend on them.
"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
IMG_DIR = ROOT / "images"

SEED = 42
N_CUSTOMERS = 1200

SEGMENTS = ["young_digital", "family", "mass_affluent", "retiree"]
PRODUCTS = ["checking_account", "savings_account", "credit_card", "personal_loan",
            "mortgage", "investment_fund", "insurance_bundle", "mobile_wallet"]
COMPLAINT_CATEGORIES = ["billing_fees", "app_technical", "wait_time", "card_issues", "communication"]


@dataclass(frozen=True)
class BusinessAssumptions:
    """Illustrative economics. Replace with the bank's real figures before using for a decision."""
    base_annual_revenue_eur: float = 60.0       # fixed yearly revenue per active customer
    revenue_rate_on_balance: float = 0.012      # extra yearly revenue as a share of average balance
    retention_horizon_years: float = 3.0        # years of revenue protected when a customer is saved
    offer_cost_eur: float = 45.0                # cost of one proactive retention contact + offer
    save_rate: float = 0.20                     # share of contacted would-be churners who are retained
    team_capacity_share: float = 0.20           # share of customers the retention team can contact
    cross_sell_margin_eur: float = 35.0         # yearly margin of one accepted cross-sell offer
    cross_sell_acceptance_baseline: float = 0.03  # acceptance rate of the generic script


ASSUMPTIONS = BusinessAssumptions()
