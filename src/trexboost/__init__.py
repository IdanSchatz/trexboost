"""trexboost: FDR-controlled feature selection with the T-Rex selector and gradient-boosted trees."""

from trexboost._selector import TRexBoostSelector
from trexboost.stability import nogueira_stability, resample_selections, stability_report

__all__ = ["TRexBoostSelector", "nogueira_stability", "resample_selections", "stability_report"]
__version__ = "0.1.0"
