"""Handle the coefficient of variation"""

from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.optimize import fsolve

from . import utils


def determine_variations(
    guidepair_residuals, genepair_names, sample_names, context_map
):
    context_map = {k.replace("-", "_"): v for k, v in context_map.items()}
    context_order = context_map.keys()
    fields = [
        f"{context}_{category}"
        for context in context_order
        for category in ["background", "boundaries"]
    ]
    Cov = namedtuple("cov", ["mean", "std", "cov", *fields])

    covs_out = calculate_covs_per_contexts(
        guidepair_residuals, genepair_names, sample_names, context_map
    )

    context_backgrounds = {}
    for name, samples in context_map.items():
        res = subset_guidepair(
            guidepair_residuals, genepair_names, sample_names, samples
        )
        _, _, background_covs, boundaries = get_background_covs(res)

        context_backgrounds[name] = (background_covs, boundaries)

    data = [df for context in context_order for df in context_backgrounds[context]]

    return Cov(*[*covs_out, *data])


def get_background_covs(guidepair_residuals, top_n=1000):
    means, stds, covs = calculate_covs(guidepair_residuals)
    top_covs = get_top_cov_by_mean(means, covs, n=top_n)

    boundaries = []
    all_means, all_stds, all_covs = [], [], []
    for _ in range(100):
        shuffled_guidepair_residuals = shuffle_ndarray(guidepair_residuals.values)
        shuffled_guidepair_residuals = pd.DataFrame(
            shuffled_guidepair_residuals,
            index=guidepair_residuals.index,
            columns=guidepair_residuals.columns,
        )

        random_means, random_stds, random_covs = calculate_covs(
            shuffled_guidepair_residuals
        )

        all_means.append(random_means)
        all_stds.append(random_stds)
        all_covs.append(random_covs)

        random_top_covs = get_top_cov_by_mean(random_means, random_covs, n=top_n)
        boundary = intersect_distributions(top_covs, random_top_covs, maxfev=10_000)
        boundaries.append(boundary)

    all_means = pd.concat(all_means, axis=1)
    all_stds = pd.concat(all_stds, axis=1)
    all_covs = pd.concat(all_covs, axis=1)
    boundaries = np.array(boundaries)

    return all_means, all_stds, all_covs, boundaries


def calculate_covs_per_contexts(
    guidepair_residuals, genepair_names, sample_names, context_map
):
    """Calculate CoVs across contexts

    Convenience function to calculate CoVs across all contexts specified.

    Paramters
    ----------
    guidepair_residuals: numpy.ndarray
        3-dimension array. genepair x 9 guidepairs x samples
    genepair_names
        Multi-index. Names for the geneapirs (correspond to the 1st dimension)
    sample_names: array
        Names of the samples (correspond to the 3rd dimension)
    context_map: dict
        Context names mapping to the cell line names. The mapping from cell
        lines to samples is done within this function (via `subset_guidepair`)

    Returns
    -------
    means, stds, covs
        Means, Standard Deviations, and Coefficients of Variations for each
        context specified as a dataframe (dimensions genepair x contexts)
    """

    covs, stds, means = [], [], []

    for name, samples in context_map.items():
        res = subset_guidepair(
            guidepair_residuals, genepair_names, sample_names, samples
        )
        mean, std, cov = calculate_covs(res, name=name)

        means.append(mean)
        stds.append(std)
        covs.append(cov)

    means = pd.concat(means, axis=1)
    stds = pd.concat(stds, axis=1)
    covs = pd.concat(covs, axis=1)

    return means, stds, covs


def calculate_covs(guide_residuals, name=None):
    """Calculate the Coefficient of Variation

    Coefficient of variation (CoV) is simply the standard deviation divided by
    the mean. I calculate the *absolute value* of the mean because my means can
    be negative. It's a small extension.
    """

    mean = guide_residuals.mean(axis=1)
    mean.name = name

    std = guide_residuals.std(axis=1)
    std.name = name

    cov = std / np.absolute(mean)
    cov.name = name

    return mean, std, cov


def get_top_cov_by_mean(means, covs, n=1000):
    """Select the top CoV by the magnitude of the mean"""

    abs_means = np.absolute(means.values.ravel())
    cutoff = np.sort(abs_means)[::-1][n]

    return covs.values.ravel()[abs_means > cutoff]


def shuffle_ndarray(array):
    shuffled_array = array.ravel().copy()
    np.random.shuffle(shuffled_array)

    return shuffled_array.reshape(array.shape)


def intersect_distributions(distribution1, distribution2, guess=None, maxfev=1000):
    f1 = gaussian_kde(distribution1)
    f2 = gaussian_kde(distribution2)

    def func(x):
        return f1(x) - f2(x)

    if guess is None:
        guess = (np.mean(distribution1) + np.mean(distribution2)) / 2
    boundary = fsolve(func, guess, maxfev=maxfev)

    return boundary


def subset_guidepair(guidepair_residuals, genepair_names, sample_names, samples_subset):
    """Get guidepair residuals based on the samples

    Retrieves the guidepairs specific to certain samples and flatten to 2-dim
    (genepair x {n_samples_subset * 9 guidepairs})

    Parameters
    ----------
    guidepair_residuals: numpy.ndarray
        3-dimension array. genepair x 9 guidepairs x samples
    genepair_names
        Multi-index. Names for the geneapirs (correspond to the 1st dimension)
    sample_names: array
        Names of the samples (correspond to the 3rd dimension)
    samples_subset: array
        Subset of `sample_names` (correspond to the 3rd dimension)

    Returns
    -------
    res : pandas.DataFrame
        Guidepair residuals that are stacked.
    """

    index = np.argwhere(np.isin(sample_names, samples_subset)).ravel()

    res = np.concatenate([guidepair_residuals[:, :, i] for i in index], axis=1)
    res = pd.DataFrame(res, index=genepair_names)

    return res


def get_context_sample_map(samples, cell_lines, contexts):
    grouped_samples = utils.get_columns(samples, cell_lines, reps=None)
    grouped_samples.update(utils.get_columns(samples, contexts, reps=None))

    return grouped_samples
