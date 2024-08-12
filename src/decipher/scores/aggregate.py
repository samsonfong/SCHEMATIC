"""Aggregate residuals into score"""

import numpy as np
import pandas as pd

from scipy.stats import ttest_ind
from collections import namedtuple
from multiprocessing import Pool
from statsmodels.stats.multitest import multipletests

from . import utils

# TODO: Make sure guidepair residuals actually keep the correct index


Scores = namedtuple("Scores", ["scores", "fdrs", "cutoffs"])


def score_interactions(residuals, groups):
    scores, cutoffs = calculate_scores_and_effect_size(
        residuals.genepair_residuals,
        groups,
    )

    fdrs = estiamte_fdrs(
        residuals.genepair_residuals, groups, residuals.guidepair_residuals
    )

    return Scores(scores, fdrs, cutoffs)


def calculate_scores_and_effect_size(
    residuals, groups, reps=(1, 2), control="AAVS", confidence=99
):
    """Given sample level residuals, calculate final score and estimate effect size

    Paramters
    ---------
    residuals: pandas.DataFrame
        A matrix of gene-pair by sample residuals that define how unexpected the
        pairwise fitness is.
    groups: list or dictionary
        A collection that defines how to aggregate the samples. If groups is a
        list, it will be converted to a dictionary where the key is the element
        and the value is a list with the element as the only item. If groups is
        a dictionary, the key is the name of the group, and the value is a list
        of sample names. A sample will be used to aggregate if the groups names
        is at least partially in the sample name.
    reps: tuple of integers
        A tuple of integers that represent the replicates. The replicate is
        expected to be denoted in the sample name as "_{replicate id}"
    """

    control_index = (
        (residuals.reset_index()[["target_a_id", "target_b_id"]] == control)
        .any(axis=1)
        .values
    )

    lower_bound = (100 - confidence) / 2
    upper_bound = 100 - lower_bound

    effect_size = {}
    scores = {}

    grouped_indices = utils.get_columns(residuals.columns, groups, reps=reps)

    for name, columns in grouped_indices.items():
        res = residuals[columns].mean(axis=1)
        res, a, b = standardize_and_threshold(
            res, control_index, lower_bound, upper_bound
        )

        effect_size[name] = (a, b)
        scores[name] = res

    scores = pd.DataFrame(scores)
    effect_size = pd.DataFrame(effect_size, index=["lower", "upper"]).T

    return scores, effect_size


def standardize_and_threshold(res, control_index, lower=0.5, upper=99.5):
    res = (res - res.mean()) / res.std()

    a = np.percentile(res.loc[control_index], lower)
    b = np.percentile(res.loc[control_index], upper)

    return res, a, b


class Constructs(object):
    # TODO: Move me out of this module

    def __init__(self, constructs, drop_extras=True):
        self.constructs = constructs

        if drop_extras:
            # Dropping guide names that are greater than 3
            keep = []
            for ind, row in enumerate(self.constructs.index):
                if int(row[2].split("-")[1]) <= 3 and int(row[4].split("-")[1]) <= 3:
                    keep.append(ind)

            self.constructs = self.constructs.iloc[keep]

        self._get_scaler()

    def get(self, geneA=None, geneB=None, column=None, scale=False, drop_level=True):
        df = self.constructs

        if geneA is not None:
            level = [ind for ind, i in enumerate(df.index.names) if i == "target_a_id"][
                0
            ]
            df = df.xs(geneA, level=level, drop_level=drop_level)

        if geneB is not None:
            level = [ind for ind, i in enumerate(df.index.names) if i == "target_b_id"][
                0
            ]
            df = df.xs(geneB, level=level, drop_level=drop_level)

        if column is not None:
            df = df[column]

        if scale:
            ntc = self.ntc_f if column is None else self.ntc_f[column]
            std = self.std if column is None else self.std[column]

            df = (df - ntc) / std

        return df

    def get_gp(self, geneA, geneB, column=None):
        return self.get(geneA, geneB, column=column, scale=True).median()

    def _get_scaler(self, ntc="nontargeting"):
        self.ntc_f = self.get(ntc, ntc).median()

        columns_to_drop = np.setdiff1d(
            self.constructs.index.names, ["target_a_id", "target_b_id"]
        )
        gp = (
            self.constructs.reset_index()
            .drop(columns=columns_to_drop)
            .groupby(["target_a_id", "target_b_id"])
            .median()
        )  # replace with reading?
        self.std = gp.std()

    def get_grouped_construct(self, sample, geneA=None, geneB=None):
        columns_to_drop = np.setdiff1d(
            self.constructs.index.names, ["target_a_id", "target_b_id"]
        )
        data = self.get(geneA, geneB, drop_level=False, column=sample, scale=True)
        grouped = (
            data.reset_index()
            .drop(columns=columns_to_drop)
            .groupby(["target_a_id", "target_b_id"])
        )
        grouped = pd.DataFrame(
            {(gA, gB): row[sample].values for (gA, gB), row in grouped}
        ).T

        return grouped


def estimate_guidepair_residuals(residuals, popt, constructs, smf, zs):
    # I expect controls to already be dropped
    constructs = Constructs(constructs, drop_extras=False)
    expand_jobs = [
        (residuals[sample], popt.loc[sample], constructs, smf[sample], zs.loc[sample])
        for sample in residuals.columns
    ]

    with Pool(None) as p:
        all_residuals = p.starmap(expand_residuals, expand_jobs)

    guidepair_residuals = np.dstack(all_residuals)

    return guidepair_residuals


def expand_residuals(residuals, popt, constructs, smf, zs):
    sample = residuals.name
    print(sample)

    residuals_sq = residuals.reset_index().pivot(
        index="target_a_id", columns="target_b_id", values=residuals.name
    )

    inner = []
    for anchor in residuals_sq.columns:
        a, b = popt.loc[anchor]
        m, s = zs.loc[anchor]

        for array in residuals_sq.index:
            # I expect constructs to be scaled already
            array_f = constructs.get(
                geneA=array, geneB=anchor, column=sample, scale=False
            )
            _smf = smf.loc[array]
            res = array_f - utils.linear_func(_smf, a, b)
            res = (res - m) / s

            inner.append(res.values)

    return inner


def estiamte_fdrs(residuals, groups, guidepair_residuals, reps=(1, 2), control="AAVS"):
    all_fdrs = []

    grouped_indices = utils.get_columns(residuals.columns, groups=groups, reps=reps)
    for name, columns in grouped_indices.items():
        columns_index = np.argwhere(
            np.isin(residuals.columns, columns)
        ).ravel()  # Convert to numerical index
        res = np.concatenate(
            [guidepair_residuals[:, :, i] for i in columns_index], axis=1
        )
        res = pd.DataFrame(res, index=residuals.index)

        fdr = _inner_estimate_fdrs(res, name, residuals.index, control=control)
        all_fdrs.append(fdr)

    all_fdrs = pd.concat(all_fdrs, axis=1)

    return all_fdrs


def _inner_estimate_fdrs(res, name, index, control="AAVS"):
    pvals = []
    for (i, j), row in res.iterrows():
        _, pval = ttest_ind(
            res.xs(control, level=0).values.ravel(),
            row.values,
            equal_var=False,
            nan_policy="omit",
        )

        pvals.append(pval)

    _, fdr, *_ = multipletests(pvals, method="fdr_bh")
    fdr = pd.Series(fdr, index=index, name=name)

    return fdr
