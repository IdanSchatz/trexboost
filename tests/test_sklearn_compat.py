import warnings

from sklearn.utils.estimator_checks import parametrize_with_checks

from trexboost import TRexBoostSelector

# small settings so the tiny datasets used by the checks can be split
_EST = TRexBoostSelector(
    n_experiments=4,
    max_dummy_multiplier=2,
    max_dummies_entered=5,
    min_samples_leaf=1,
    max_rounds=200,
    random_state=0,
)


@parametrize_with_checks([_EST])
def test_sklearn_compatible(estimator, check):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check(estimator)
