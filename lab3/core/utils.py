import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression


def get_model_formula(model: LinearRegression) -> str:
    formula = f"{model.intercept_}"
    for coefficient, feature in zip(model.coef_, model.feature_names_in_):
        if coefficient > 0:
            formula += f" + {coefficient} * {feature}"
            continue
        formula += f" - {abs(coefficient)} * {feature}"
    return formula


def foster_stuart_criterion(x):
    n = len(x)
    p = np.ones(n - 1)
    q = np.ones(n - 1)

    for i in range(1, n):
        for j in range(i):
            if x[i] >= x[j]:
                p[i - 1] = 0
            if x[i] <= x[j]:
                q[i - 1] = 0
            if q[i - 1] + p[i - 1] == 0:
                break
    t = abs(np.sum(p - q) / np.sqrt(2 * np.sum(1 / np.arange(2, n))))
    return t
