from .pipeline import calculate_interactions
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from multiprocessing import Pool, cpu_count
from collections import namedtuple
from functools import partial

from . import utils, aggregate


Residuals = namedtuple(
    "Residuals",
    ["genepair_residuals", "parameters", "expected", "scalers", "guidepair_residuals"],
)


def score_residuals(fitnesses, pre_standardize=False, post_standardize=True):
    residuals, popt, expected, scalers = score_all_residuals(
        fitnesses.genepair,
        fitnesses.smf_a,
        pre_standardize=pre_standardize,
        post_standardize=post_standardize,
        scale_residuals=True,
        mirror=True,
    )

    # TODO: Need to include options to standardize and scale
    guidepair_residuals = aggregate.estimate_guidepair_residuals(
        residuals, popt, fitnesses.construct, fitnesses.smf_a, scalers
    )

    residuals_results = Residuals(
        residuals, popt, expected, scalers, guidepair_residuals
    )

    return residuals_results


def interactions_per_array(
    l2e,
    smf_map,
    gA_col="probe_a_id",
    gB_col="probe_b_id",
    score_col="fitness",
    fix_intercept=False,
    smf_intercept_map=None,
    arrays_to_skip=None,
):
    """Calculates genetic interactions for each array

    Paramters
    ---------
    l2e : pandas DataFrame
        DataFrame that contains the observed pairwise fitness. The DataFrame
        will be pivoted
    smf : dict
        Maps the identifier to its single mutant fitness
    gA_col : str
        Column name of the l2e DataFrame that names the first position
    gB_col : str
        Column name of the l2e DataFrame that names the second position
    score_col : str
        Column name of the l2e DataFrame that contains the fitness
    fix_intercept : bool
        If True, the y-intercept will be fixed the the smf defined in `smf_intercept_map`
    arrays_to_skip : iterable
        Identify any guides that should be skipped when scoring interactions
    """

    if arrays_to_skip is None:
        arrays_to_skip = []

    if not np.all(np.isin([gA_col, gB_col, score_col], l2e.columns)):
        l2e = l2e.reset_index()

    sq_l2e = l2e.pivot(gA_col, gB_col, score_col)

    control = np.array([smf_map[i] for i in sq_l2e.index])

    results = []
    regression_params = []
    array_names = []
    for g in sq_l2e.columns:
        if g in arrays_to_skip:
            continue

        array_names.append(g)
        observed = sq_l2e[g].values

        # Remove non-finite values
        valid_index = np.argwhere(np.isfinite(observed)).ravel()
        observed = observed[valid_index]
        control_ = control[valid_index]
        index = sq_l2e.index[valid_index]

        if fix_intercept:
            y_intercept = smf_intercept_map[g]
            model = utils.linear_func_factory(y_intercept)
        else:
            model = utils.linear_func

        popt, _ = curve_fit(model, control_, observed)

        expected = model(control_, *popt)
        interactions = observed - expected

        interactions = (interactions - np.mean(interactions)) / np.std(interactions)

        if fix_intercept:
            popt = np.hstack([popt, y_intercept])

        regression_params.append(popt)

        n = len(interactions)
        df = pd.DataFrame(
            [
                index,
                [g] * n,
                control_,
                expected,
                observed,
                interactions,
                [score_col] * n,
            ],
            index=[
                gA_col,
                gB_col,
                "Control",
                "Expected",
                "Observed",
                "Interactions",
                "Name",
            ],
        ).T
        results.append(df)

    regression_params = pd.DataFrame(
        np.vstack(regression_params),
        index=array_names,
        columns=["slope", "y-intercepts"],
    )

    results = pd.concat(results, axis=0).set_index([gA_col, gB_col])
    results = results.astype(
        {
            "Control": float,
            "Expected": float,
            "Interactions": float,
            "Observed": float,
            "Name": str,
        }
    )

    return results, regression_params


def interactions_by_min(
    l2e, smf_map, gA_col="probe_a_id", gB_col="probe_b_id", score_col="fitness"
):
    """Calculates genetic interactions based on the min definition

    Paramters
    ---------
    l2e : pandas DataFrame
        DataFrame that contains the observed pairwise fitness. The DataFrame
        will be pivoted
    smf_map : dict of dict
        Dictionary of two dictionaries. First (key = 'A') maps the single gene
        fitness of the first position. Second (key = 'B') maps the single gene
        fitness of the second position.
    gA_col : str
        Column name of the l2e DataFrame that names the first position
    gB_col : str
        Column name of the l2e DataFrame that names the second position
    score_col : str
        Column name of the l2e DataFrame that contains the fitness
    """

    if not np.all(np.isin([gA_col, gB_col, score_col], l2e.columns)):
        l2e = l2e.reset_index()

    fA = l2e[gA_col].map(smf_map["A"]).values
    fB = l2e[gB_col].map(smf_map["B"]).values

    expected = np.vstack([fA, fB])
    expected = np.min(expected, axis=0)
    observed = l2e[score_col].values

    interactions = observed - expected
    interactions = (interactions - np.mean(interactions)) / np.std(interactions)

    results = l2e[[gA_col, gB_col]]
    results = results.assign(
        Expected=expected,
        Observed=observed,
        Interactions=interactions,
        Name=[score_col] * len(fA),
    )

    results = results.set_index([gA_col, gB_col])

    return results.astype(
        {"Expected": float, "Interactions": float, "Observed": float, "Name": str}
    )


def stabilize_variance(scores, expected):
    scores = scores.reshape(-1, 1)
    expected = expected.reshape(-1, 1)

    tree = KDTree(expected)
    indices = tree.query(expected, return_distance=False, k=200)

    std = np.std(scores[indices], axis=1)
    scores = scores / std

    return scores


def t_score_aggregate(
    interactions,
    control="nontargeting",
    score_col="scores",
    gA_col="target_a_id",
    gB_col="target_b_id",
):
    control_interactions = interactions.loc[
        (interactions[gA_col] == control) | (interactions[gB_col] == control), score_col
    ].values

    u_ctrl = np.median(control_interactions)
    v_ctrl = np.var(control_interactions)
    n_ctrl = control_interactions.shape[0]

    gi_t = []
    for (g1, g2), df in interactions.groupby([gA_col, gB_col]):
        u_exp = np.median(df[score_col].values)
        v_exp = np.var(df[score_col].values)
        n_exp = df.shape[0]

        s_var = v_exp * (n_exp - 1) + ((v_ctrl * (n_ctrl - 1)) / (n_exp + n_ctrl - 2))
        g = (u_exp - u_ctrl) / np.sqrt((s_var / n_exp) + (s_var / n_ctrl))

        gi_t.append([g1, g2, g])

    gi_t = pd.DataFrame(gi_t, columns=[gA_col, gB_col, "Interactions_t"])
    gi_t["Interactions_t"] = (
        gi_t["Interactions_t"] - gi_t["Interactions_t"].mean()
    ) / gi_t["Interactions_t"].std()

    return gi_t


def mwu_aggregate(
    interactions,
    control="nontargeting",
    score_col="scores",
    gA_col="target_a_id",
    gB_col="target_b_id",
    multipletests_method="fdr_bh",
):
    control_interactions = interactions.loc[
        (interactions[gA_col] == control) | (interactions[gB_col] == control), score_col
    ].values

    results = []
    for (g1, g2), df in interactions.groupby([gA_col, gB_col]):
        stat, pval = mannwhitneyu(
            control_interactions, df[score_col].values, alternative="two-sided"
        )

        results.append([g1, g2, stat, pval])

    results = pd.DataFrame(results, columns=[gA_col, gB_col, "stat", "pval"])

    corrected = multipletests(
        results["pval"].values,
        method=multipletests_method,
        is_sorted=False,
        returnsorted=False,
    )

    results["pval"] = corrected[1]

    return results


def calculate_residuals(x, y, func=utils.linear_func, **kwargs):
    """Calculates the residuals after linear regression"""

    residuals_ = np.full(x.shape, np.nan)

    data = np.vstack([x, y])
    nan_index = np.isnan(data).any(axis=0)

    x = x[~nan_index]
    y = y[~nan_index]

    popt, _ = curve_fit(func, x, y, **kwargs)
    y_hat = func(x, *popt)

    residuals = y - y_hat
    residuals_[~nan_index] = residuals

    return popt, y_hat, residuals_


def interactions_with_resampling(xdata, ydata, n_resamples=1000):
    _, gB_size, ntpts, nsamples = ydata.shape

    interactions_outer = []
    popt_outer = []
    for gene_index in range(gB_size):
        interactions_middle = []
        popt_middle = []
        for t in range(ntpts):
            x_sampled = xdata[:, t, :]
            y_sampled = ydata[:, gene_index, t, :]

            interactions_sampled = []
            popt_sampled = []
            for _ in range(n_resamples):
                index = np.random.randint(0, nsamples, size=(x_sampled.shape[0], 1))

                popt, _, res = calculate_residuals(
                    np.take_along_axis(x_sampled, index, 1).squeeze(),
                    np.take_along_axis(y_sampled, index, 1).squeeze(),
                )
                interactions_sampled.append(res)
                popt_sampled.append(popt)

            interactions_middle.append(np.stack(interactions_sampled, axis=1))
            popt_middle.append(np.stack(popt_sampled, axis=1))

        interactions_outer.append(np.stack(interactions_middle, axis=1))
        popt_outer.append(np.stack(popt_middle, axis=1))

    interactions_outer = np.stack(interactions_outer, axis=1)
    popt_outer = np.stack(popt_outer, axis=1)

    return interactions_outer, popt_outer


def interactions_without_resampling(xdata, ydata):
    _, gB_size, ntpts = ydata.shape

    outer = []
    for gene_index in range(gB_size):
        middle = []
        for t in range(ntpts):
            x = xdata[:, t]
            y = ydata[:, gene_index, t]

            popt, y_hat, res = calculate_residuals(x, y)
            middle.append(res)

        middle = np.stack(middle, axis=1)
        outer.append(middle)

    outer = np.stack(outer, axis=1)

    return outer


def process_raw_scores(raw_scores, name, force_symmetric=True):
    raw_ints = np.median(raw_scores.interactions, axis=-1)  # Median across resamplings
    raw_ints = raw_ints.reshape(-1, raw_ints.shape[-1], order="C")
    genepairs = [
        (i, j)
        for i in raw_scores.pivoted_index.index
        for j in raw_scores.pivoted_index.columns
    ]
    genepairs = pd.MultiIndex.from_tuples(genepairs)

    scores = np.percentile(raw_ints, 50, axis=-1)  # Median across timepoints
    scores = (scores - np.mean(scores)) / np.std(scores)

    background = utils.get_background_values(raw_ints, axis=1)
    left_fdr = utils.estimate_empirical_fdr(scores, background, side="left")
    right_fdr = utils.estimate_empirical_fdr(scores, background, side="right")

    df = pd.DataFrame(
        np.vstack([scores, left_fdr, right_fdr]).T,
        index=genepairs,
        columns=[f"{name}", f"{name}-left_fdr", f"{name}-right_fdr"],
    )
    df.index.names = ["target_a_id", "target_b_id"]

    if force_symmetric:
        df = utils.make_symmetric(
            df.reset_index(), "target_a_id", "target_b_id", [name]
        )

    return df


def process_raw_scores_without_resampling(
    residuals, pivoted_index, name, force_symmetric=True
):
    raw_ints = residuals.reshape(-1, residuals.shape[-1], order="C")
    genepairs = [(i, j) for i in pivoted_index.index for j in pivoted_index.columns]
    genepairs = pd.MultiIndex.from_tuples(genepairs)

    scores = np.percentile(raw_ints, 50, axis=-1)  # Median across timepoints
    scores = (scores - np.mean(scores)) / np.std(scores)

    background = utils.get_background_values(raw_ints, axis=1)
    left_fdr = utils.estimate_empirical_fdr(scores, background, side="left")
    right_fdr = utils.estimate_empirical_fdr(scores, background, side="right")

    df = pd.DataFrame(
        np.vstack([scores, left_fdr, right_fdr]).T,
        index=genepairs,
        columns=[f"{name}", f"{name}-left_fdr", f"{name}-right_fdr"],
    )
    df.index.names = ["target_a_id", "target_b_id"]

    if force_symmetric:
        df = utils.make_symmetric(
            df.reset_index(), "target_a_id", "target_b_id", [name]
        )

    return df


def calculate_scores_from_residuals(residuals, index, columns):
    reses = []
    for i in range(residuals.shape[2]):
        res = residuals[:, :, i]
        res = pd.DataFrame(res, index=index, columns=columns)
        aavs = res["AAVS"].std()
        res = (
            res.reset_index()
            .melt(id_vars="target_a_id", value_name=i)
            .set_index(["target_a_id", "target_b_id"])
        )
        res = (res - res.mean(axis=0)) / aavs
        reses.append(res)

    reses = pd.concat(reses, axis=1)
    reses = reses.mean(axis=1)

    return reses


def standardize_residuals(residuals, index, columns):
    reses = []
    for i in range(residuals.shape[2]):
        res = residuals[:, :, i]
        res = (res - res.mean(axis=0)) / res.std(axis=0)
        res = pd.DataFrame(res, index=index, columns=columns)
        res = (
            res.reset_index()
            .melt(id_vars="target_a_id", value_name=i)
            .set_index(["target_a_id", "target_b_id"])
        )
        reses.append(res)

    reses = pd.concat(reses, axis=1)
    reses = reses.mean(axis=1)

    return reses


def score_all_residuals(
    genepair_fitness,
    smf,
    geneA_name="target_a_id",
    geneB_name="target_b_id",
    ncpus=-1,
    pre_standardize=False,
    post_standardize=False,
    scale_residuals=True,
    mirror=False,
    **kwargs,
):
    if ncpus == -1:
        ncpus = min(len(genepair_fitness.columns), cpu_count())

    _score_sample_residuals = partial(
        score_sample_residuals,
        pre_standardize=pre_standardize,
        post_standardize=post_standardize,
        scale_residuals=scale_residuals,
        mirror=mirror,
        **kwargs,
    )

    pool = Pool(ncpus)
    args = []
    for column in genepair_fitness.columns:
        sq = (
            genepair_fitness[column]
            .reset_index()
            .pivot(index=geneA_name, columns=geneB_name, values=column)
        )
        args.append((sq, smf[column], column))

    container = pool.starmap(_score_sample_residuals, args)
    pool.close()

    residuals = []
    popt = []
    yhat = []
    zs = []
    for residuals_, popt_, yhat_, z_ in container:
        residuals.append(residuals_)
        popt.append(popt_)
        yhat.append(yhat_)
        zs.append(z_)

    residuals = pd.concat(residuals, axis=1)
    popt = pd.concat(popt, axis=0)
    yhat = pd.concat(yhat, axis=1)
    zs = pd.concat(zs, axis=0)

    return residuals, popt, yhat, zs


def score_sample_residuals(
    sq_genepair_fitness,
    smf,
    sample_name,
    pre_standardize=False,
    post_standardize=True,
    scale_residuals=True,
    mirror=True,
    **kwargs,
):
    """Private method to score residuals (made for multiprocessing)"""

    x = smf.values
    smf_names = smf.index

    geneA_name = sq_genepair_fitness.index.name
    geneB_name = sq_genepair_fitness.columns.name

    residuals = []
    popt = []
    yhat = []
    zs = []
    for anchor in sq_genepair_fitness.columns:
        y = sq_genepair_fitness.loc[smf_names, anchor].values
        if pre_standardize:
            y = (y - y.mean()) / y.std()

        _popt, _yhat, _residuals = calculate_residuals(x, y, **kwargs)

        zs.append([np.mean(_residuals), np.std(_residuals)])

        if scale_residuals:
            _residuals = (
                _residuals / _popt[0]
            )  # this should have no effect on residuals when paired with post_standardize

        if post_standardize:
            _residuals = (_residuals - np.mean(_residuals)) / np.std(_residuals)

        residuals.append(_residuals)
        popt.append(_popt)
        yhat.append(_yhat)

    residuals = pd.DataFrame(
        np.vstack(residuals).T,
        columns=sq_genepair_fitness.columns,
        index=smf_names,
    )
    residuals.index.name = geneA_name

    if mirror:
        overlap = residuals.index.intersection(residuals.columns)
        sq = residuals.loc[overlap, overlap]
        sq = (sq + sq.T) / 2
        residuals.loc[overlap, overlap] = sq

    residuals.columns.name = geneB_name

    residuals = residuals.reset_index().melt(id_vars=geneA_name, value_name=sample_name)
    residuals = residuals.set_index([geneA_name, geneB_name])

    index = pd.MultiIndex.from_tuples(
        [(sample_name, i) for i in sq_genepair_fitness.columns]
    )
    popt = pd.DataFrame(np.vstack(popt), index=index, columns=["a", "b"])

    zs = pd.DataFrame(np.vstack(zs), index=index, columns=["mean", "std"])

    yhat = pd.DataFrame(
        np.vstack(yhat).T, columns=sq_genepair_fitness.columns, index=smf_names
    )
    yhat.index.name = geneA_name
    yhat = (
        yhat.reset_index()
        .melt(id_vars=geneA_name, value_name=sample_name)
        .set_index([geneA_name, geneB_name])
    )

    return residuals, popt, yhat, zs


def score_alternative_model(genepair, smf_a, smf_b, groups, model="additive"):
    if model == "additive":
        expected = smf_a.values[:, np.newaxis, :] + smf_b.values[np.newaxis, :, :]
    elif model == "minimum":  # Mostly for negative interactions
        expected = np.minimum(
            smf_a.values[:, np.newaxis, :], smf_b.values[np.newaxis, :, :]
        )
    elif model == "maximum":  # Mostly for positive interactions
        expected = np.maximum(
            smf_a.values[:, np.newaxis, :], smf_b.values[np.newaxis, :, :]
        )
    else:
        raise NotImplementedError

    additive_model = genepair - expected.reshape(-1, 47)

    columns = utils.get_columns(genepair.columns, groups)

    scores = []
    for name, cols in columns.items():
        df = additive_model[cols].median(axis=1)
        df.name = name

        scores.append(df)

    scores = pd.concat(scores, axis=1)

    # Determining effect size
    aavs = pd.concat([scores.xs("AAVS", level=0), scores.xs("AAVS", level=1)], axis=0)
    cutoff = np.percentile(aavs, 0.5, axis=0)
    cutoff = np.minimum(cutoff, 0)

    return scores, cutoff
