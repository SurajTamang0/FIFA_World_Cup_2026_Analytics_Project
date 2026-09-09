from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def descriptive_statistics(series: pd.Series) -> dict:
    """
    Calculate descriptive statistics for a numeric pandas Series.
    Missing values are removed before calculation.
    """

    clean = pd.to_numeric(series, errors="coerce").dropna()

    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std(ddof=1)),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def mean_confidence_interval(
    series: pd.Series,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """
    Calculate a confidence interval for the population mean
    using the t distribution.

    Returns:
        mean, lower_bound, upper_bound
    """

    clean = pd.to_numeric(series, errors="coerce").dropna()

    n = len(clean)

    if n < 2:
        raise ValueError(
            "At least two observations are required "
            "to calculate a confidence interval."
        )

    mean = clean.mean()
    standard_error = stats.sem(clean)

    alpha = 1 - confidence
    critical_value = stats.t.ppf(
        1 - alpha / 2,
        df=n - 1,
    )

    margin_of_error = critical_value * standard_error

    lower = mean - margin_of_error
    upper = mean + margin_of_error

    return float(mean), float(lower), float(upper)


def welch_t_test(
    group_a: pd.Series,
    group_b: pd.Series,
) -> dict:
    """
    Perform an independent two-sample Welch t-test.

    Welch's test is used because it does not assume
    equal population variances.
    """

    a = pd.to_numeric(group_a, errors="coerce").dropna()
    b = pd.to_numeric(group_b, errors="coerce").dropna()

    if len(a) < 2 or len(b) < 2:
        raise ValueError(
            "Each comparison group must contain "
            "at least two observations."
        )

    result = stats.ttest_ind(
        a,
        b,
        equal_var=False,
        nan_policy="omit",
    )

    return {
        "group_a_n": len(a),
        "group_b_n": len(b),
        "group_a_mean": float(a.mean()),
        "group_b_mean": float(b.mean()),
        "mean_difference": float(a.mean() - b.mean()),
        "t_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def iqr_outlier_summary(series: pd.Series) -> dict:
    """
    Identify potential outliers using the 1.5 x IQR rule.
    This is used as an assumption-diagnostic aid,
    not as an automatic rule for deleting observations.
    """

    clean = pd.to_numeric(series, errors="coerce").dropna()

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = clean[
        (clean < lower_bound)
        | (clean > upper_bound)
    ]

    return {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outlier_count": int(len(outliers)),
        "outlier_percentage": float(
            len(outliers) / len(clean) * 100
        ) if len(clean) > 0 else 0.0,
    }