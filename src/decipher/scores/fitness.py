"""Module to infer fitnesses"""

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from . import utils

from collections import namedtuple


Fitnesses = namedtuple("Fitnesses", ["construct", "genepair", "smf_a", "smf_b"])


def score_fitness(counts_file, essential_genes):
    """Score all fitnesses"""

    counts = pd.read_csv(counts_file, index_col=[0, 1, 2, 3, 4])
    counts = filter_constructs(counts, log2_cutoff=-23)

    construct_fitness = log2enrichment_df(
        counts,
        "plasmid",
        counts.columns,
        pseudocount=1,
    ).drop(columns=["plasmid"])

    ntc_ntc_fitness, construct_fitness = utils.pop_controls(construct_fitness)

    genepair_fitness = calculate_genepair_fitness(construct_fitness)
    genepair_fitness, f_ntc, std_f = standardize_genepair_fitness(genepair_fitness)
    genepair_fitness = remove_samples(genepair_fitness, essential_genes)

    construct_fitness = (construct_fitness - f_ntc) / std_f

    smf = calculate_smf(genepair_fitness, axis=1)
    smf_b = calculate_smf(genepair_fitness, axis=0)

    return Fitnesses(construct_fitness, genepair_fitness, smf, smf_b)


def log2enrichment(
    start_counts,
    end_counts,
    normalize=True,
    normalize_axis=0,
    pseudocount=1,
):
    """Log2 Enrichment

    Note
    ----
    If nans are in the counts, they are ignored (essentially treated as 0).

    Paramters
    ---------
    start_counts : numpy.ndarray
       Starting counts for each construct. If one-dimensional, it will be
       reshaped to match the dimensions of `end_counts`. If `end_counts` has
       more than one dimension, start_counts will be reshaped to (n, 1, ...)
       where n is the length of start_counts.
    end_counts : numpy.ndarray
        End counts for each construct
    normalize : bool
        If True, both start_counts and end_counts will be divided by the sum
        of each array over the first dimension
    normalize_axis : int
        The axis to normalize the data over
    pseudocount : int
        A count added to each element of both start_counts and end_counts array

    Returns
    -------
    l2e : numpy.ndarray(1-dimensional)
        The log2 enrichment of the constructs
    """

    start_counts = np.asanyarray(start_counts).astype(float) + pseudocount
    end_counts = np.asanyarray(end_counts).astype(float) + pseudocount

    # Reshaping start_counts if it is one-dimensional
    if start_counts.ndim == 1:
        shape = [1] * end_counts.ndim
        shape[0] = start_counts.shape[0]
        start_counts = start_counts.reshape(shape)

    if normalize:
        start_counts = utils.normalize(start_counts, axis=normalize_axis)
        end_counts = utils.normalize(end_counts, axis=normalize_axis)

    l2e = np.ma.log2(np.ma.divide(end_counts, start_counts)).filled(np.nan)

    return l2e


def log2enrichment_df(
    counts,
    start_col,
    end_col,
    pseudocount=0,
    average=False,
    result_columns=None,
):
    """Calculate construct fitness using log2 enrichment

    If start of end column contains a list, the other will be expanded. If one
    of the column name is a list, each column will be iterated across and the
    results will be averaged.

    Parameters
    ----------
    counts : pandas.DataFrame
        DataFrame that contains the counts (ints or floats)
    start_col : str or list of str
        Identifier of the starting column(s)
    end_col : str or list of str
        Identifier of the end column(s)
    pseudocount : int
        Add constant to the rows before log transformation
    result_columns : iterable
        If available, the result columns will be named accordingly. Otherwise,
        it will have the same names as end_col. If the results were averaged,
        the column name will just be "log2 enrichment" by default.

    Returns
    -------
    construct_fitness : pd.DataFrame
        Dataframe containing the log2 enrichment
    """

    try:
        start_col, end_col = utils.expand_collections(start_col, end_col)
    except ValueError:
        raise ValueError("start_col and end_col needs to be the same length iterable.")

    utils.validate_df_dtypes(counts, np.number, pure=True)

    l2e = log2enrichment(
        counts[start_col].values,
        counts[end_col].values,
        normalize=True,
        pseudocount=pseudocount,
    )

    l2e = pd.DataFrame(l2e, index=counts.index)

    if result_columns is None:
        result_columns = end_col

    l2e.columns = result_columns

    return l2e


def area_under_log2enrichment(
    start_counts,
    end_counts,
    x,
    normalize=True,
    normalize_axis=0,
    pseudocount=1,
    sort=True,
    axis=1,
):
    """Calculates the area under log2 enrichment curve

    Paramters
    ---------
    start_counts : numpy.ndarray
        Starting counts. See `log2enrichment` for more details
    end_counts : numpy.ndarray
        End counts. See `log2enrichment` for more details. `end_counts` must
        have at least 2 dimensions
    x : iterable
        A one-dimensional array-like input
    normalize : bool
        If True, both start_counts and end_counts will be divided by the sum
        of each array over the first dimension
    normalize_axis : int
        The axis to normalize the data over
    pseudocount : int
        A count added to each element of both start_counts and end_counts array
    sort : bool
        If True, x will be sorted and `end_counts` will be sorted along `axis`.
        `start_counts` does not get sorted regardless.

    Returns
    -------
    areas : numpy.ndarray
        The area under the log2 enrichment curve
    """

    x = np.asarray(x)
    end_counts = np.asarray(end_counts)
    if x.ndim != 1:
        raise ValueError("Expected x to be one-dimensional!")

    if end_counts.ndim <= 1:
        raise ValueError("Expected end_counts to have at least 2 dimensions")

    if sort:
        index = np.argsort(x)
        end_counts = np.take(end_counts, index, axis=axis)
        x = x[index]

    l2e = log2enrichment(
        start_counts,
        end_counts,
        normalize=normalize,
        normalize_axis=normalize_axis,
        pseudocount=pseudocount,
    )
    areas = np.trapz(l2e, x=x, axis=axis)

    return areas


def area_under_log2enrichment_df(
    counts,
    start_col,
    end_col,
    normalize=True,
    normalize_axis=0,
    pseudocount=1,
    return_cols=None,
):
    """Calculates the area under log2 enrichment curve for a dataframe

    This function DOES NOT parse out replicate. This might produce the wrong
    answer.

    Paramters
    ---------
    counts : pandas.DataFrame
        DataFarme containing counts.
    start_col : iterable
        Columns for starting counts.
    end_counts : iterable
        Columns for end counts.
    normalize : bool
        If True, both start_counts and end_counts will be divided by the sum
        of each array over the first dimension
    normalize_axis : int
        The axis to normalize the data over
    pseudocount : int
        A count added to each element of both start_counts and end_counts array

    Returns
    -------
    areas : numpy.ndarray
        The area under the log2 enrichment curve
    """

    try:
        start_col, end_col = utils.expand_collections(start_col, end_col)
    except ValueError:
        raise ValueError("start_col and end_col needs to be the same length iterable.")

    utils.validate_df_dtypes(counts, np.number, pure=True)

    x = utils.extract_time(end_col)
    if len(np.unique(x)) < len(x):
        warnings.warn(
            "Duplicate timepoints found! This might produce the wrong results!"
        )

    mat = area_under_log2enrichment(
        counts[start_col].values,
        counts[end_col].values,
        x,
        normalize=normalize,
        normalize_axis=normalize_axis,
        pseudocount=pseudocount,
        sort=True,
        axis=1,
    )

    ret = pd.DataFrame(mat, index=counts.index)

    if return_cols is not None:
        ret.columns = return_cols

    return ret


def filter_constructs(counts, log2_cutoff=-23):
    """Remove constructs that are less abundant than a cut off"""

    plasmid_abundance = np.log2((counts["plasmid"] + 1) / (counts["plasmid"] + 1).sum())
    failing_index = (plasmid_abundance < log2_cutoff).values

    counts.loc[failing_index] = np.nan

    return counts


def aggregate_across_array(
    scores,
    score_col,
    gA_col="target_a_id",
    gB_col="target_b_id",
    agg_method=np.nanmedian,
):
    """Aggregate guide level fitness to gene level by taking the center across all

    Paramters
    ---------
    scores : pandas.DataFrame
        DataFrame that holds the data
    score_col : str
        The column that holds the scores
    gA_col : str
        Name of the column that identifies the target (not guide) of position A
    gB_col : str
        Name of the column that identifies the target (not guide) of position B
    agg_method : callable
        Method to integrate an array into a single value

    Returns
    -------
    aggregated : pandas DataFrame
        DataFrame where each row corresponds to the number of unique targets in
        both positions. There are three columns (Gene, `score_col`, Orientation).
    """

    if gA_col not in scores.columns or gB_col not in scores.columns:
        scores = scores.reset_index()

    gA = scores[gA_col].unique()
    gB = scores[gB_col].unique()

    aggregated = []
    for gAB, orientation, column in zip([gA, gB], ["A", "B"], [gA_col, gB_col]):
        for g in gAB:
            data = scores.loc[scores[column] == g, score_col]
            fitness = agg_method(data)

            aggregated.append([g, fitness, orientation])

    aggregated = pd.DataFrame(aggregated, columns=["Gene", score_col, "Orientation"])

    return aggregated


def aggregate_across_controls(
    scores,
    score_col,
    gA_col="target_a_id",
    gB_col="target_b_id",
    ntc_cols=("nontargeting-1", "nontargeting-2", "nontargeting-3"),
    agg_method=np.nanmedian,
):
    """Aggregate guide level fitness to gene level by taking the center across controls

    Paramters
    ---------
    scores : pandas.DataFrame
        DataFrame that holds the data
    score_col : str
        The column that holds the scores
    gA_col : str
        Name of the column that identifies the target (not guide) of position A
    gB_col : str
        Name of the column that identifies the target (not guide) of position B
    ntc_cols : iterable of str
        An iterable of str that identifies the control columns. It will be
        converted into a list
    agg_method : callable
        Method to integrate an array into a single value

    Returns
    -------
    aggregated : pandas DataFrame
        DataFrame where each row corresponds to the number of unique targets in
        both positions. There are three columns (Gene, `score_col`, Orientation).
    """

    ntc_cols = list(ntc_cols)

    if gA_col not in scores.columns or gB_col not in scores.columns:
        scores = scores.reset_index()

    gA = scores[gA_col].unique()
    gB = scores[gB_col].unique()

    aggregated = []
    for gAB, orientation, column in zip([gA, gB], ["A", "B"], [gA_col, gB_col]):
        for g in gAB:
            if column == gA_col:
                other_col = gB_col
            else:
                other_col = gA_col

            data = scores.loc[
                (scores[column] == g) & (scores[other_col].isin(ntc_cols)), score_col
            ]
            fitness = agg_method(data)

            aggregated.append([g, fitness, orientation])

    aggregated = pd.DataFrame(aggregated, columns=["Gene", score_col, "Orientation"])

    return aggregated


def center_fitness_by_nontargeting(
    guidepair_fitness,
    ntc_name="nontargeting",
    geneA_name="target_a_id",
    geneB_name="target_b_id",
):
    """Centers fitness by the nontargeting control pairs

    guidepair_fitness is expected to *only* include numerical values!
    """

    ind = np.argwhere(
        np.isin(guidepair_fitness.index.names, [geneA_name, geneB_name])
    ).ravel()

    if len(ind) == 0:
        raise ValueError(
            f"{geneA_name} and {geneB_name} cannot be found " "in index names"
        )

    index = np.asarray([list(row) for row in guidepair_fitness.index])
    ntc_index = np.argwhere(np.all(index[:, ind] == ntc_name, axis=1)).ravel()

    ntc_fitness = guidepair_fitness.iloc[ntc_index].median(axis=0)

    guidepair_fitness = guidepair_fitness - ntc_fitness

    return guidepair_fitness


def aggregate_guidepair(
    guidepair_fitness,
    geneA_name="target_a_id",
    geneB_name="target_b_id",
    agg=np.nanmedian,
):
    """Calculates genepair fitness by averaging across guidepairs"""

    if (
        geneA_name not in guidepair_fitness.columns
        or geneB_name not in guidepair_fitness.columns
    ):
        guidepair_fitness = guidepair_fitness.reset_index()

    genepair_fitness = guidepair_fitness.groupby([geneA_name, geneB_name]).agg(agg)

    return genepair_fitness


def aggregate_guidepair_with_resampling(
    guidepair_fitness,
    columns,
    geneA_name="target_a_id",
    geneB_name="target_b_id",
    agg=np.nanmedian,
):
    """Calculates genepair fitness by averaging across guidepairs"""

    if not all(guidepair_fitness.columns.isin([geneA_name, geneB_name])):
        guidepair_fitness = guidepair_fitness.reset_index()

    agg_results = []
    samples = []
    index = []
    for genes, df in guidepair_fitness.groupby(["target_a_id", "target_b_id"]):
        df = df[columns]
        data = df.values
        index.append(genes)

        samples_inner = []
        results_inner = []
        for ind, column in enumerate(columns):
            values = data[:, ind]
            resampled = np.random.choice(
                values, size=(values.shape[0], 1000), replace=True
            )
            resampled_medians = np.median(resampled, axis=0)

            samples_inner.append(resampled_medians)
            results_inner.append(np.median(values))

        samples.append(np.array(samples_inner))
        agg_results.append(np.array(results_inner))

    samples = np.stack(samples, axis=0)
    agg_results = np.stack(agg_results, axis=0)

    index = pd.MultiIndex.from_tuples(index)
    agg_results = pd.DataFrame(agg_results, index=index, columns=columns)

    return agg_results, samples


def get_ab_ba_fitness(data, gA_column, gB_column, value_column):
    """Gets fitness that are target the same genepair"""

    column_names = (gA_column, gB_column, value_column)
    if np.any([i not in data.columns for i in column_names]):
        data = data.reset_index()

    square = data.pivot(*column_names)
    overlap = square.index.intersection(square.columns)

    square = square.loc[overlap, overlap]
    square_values = square.values

    names = []
    x = []
    y = []
    for i, (index, row) in enumerate(square.iterrows()):
        for j, (column, value) in enumerate(row.iteritems()):
            if i >= j:
                continue

            names.append((index, column))
            x.append(value)
            y.append(square_values[j, i])

    return x, y, names


def smf_by_agg(data, axis=1, method="median"):
    if method == "median":
        return np.median(data, axis=axis)
    else:
        raise NotImplementedError


def calculate_smf(data, gA_name="target_a_id", gB_name="target_b_id", axis=1):
    smf = []
    for column in data.columns:
        f = (
            data[column]
            .reset_index()
            .pivot(index=gA_name, columns=gB_name, values=column)
        )
        smf.append(f.median(axis=axis))

    smf = pd.concat(smf, axis=1)
    smf.columns = data.columns

    return smf


def calculate_genepair_fitness(
    construct_fitness,
    geneA_name="target_a_id",
    geneB_name="target_b_id",
):
    columns_to_drop = np.setdiff1d(
        construct_fitness.index.names, [geneA_name, geneB_name]
    )

    genepair_fitness = (
        construct_fitness.reset_index()
        .drop(columns=columns_to_drop)
        .groupby([geneA_name, geneB_name])
        .median()
    )

    return genepair_fitness


def standardize_genepair_fitness(genepair_fitness, ntc="nontargeting"):
    """Scale genepair fitness by nontargeting and variance"""

    f_ntc = genepair_fitness.loc[ntc, ntc]
    std = genepair_fitness.std(axis=0)

    centered_fitness = genepair_fitness - f_ntc
    std_fitness = centered_fitness / std

    return std_fitness, f_ntc, std


def remove_samples(genepair_fitness, common_essentials, min_score=-1):
    """Remove samples if the genepair fitness of essential gene pairs does not meet a threshold"""

    essential_index = (
        genepair_fitness.reset_index()[["target_a_id", "target_b_id"]]
        .isin(common_essentials)
        .all(axis=1)
        .values
    )

    passing_samples = (
        genepair_fitness.loc[essential_index].median() < min_score
    ).values
    genepair_fitness = genepair_fitness.loc[:, passing_samples]

    return genepair_fitness


def calibrate_fitnesses(fitnesses, parameters):
    calibrated = [[], [], [], []]

    for sample in fitnesses.genepair.columns:
        try:
            gp = (
                fitnesses.genepair[sample]
                .reset_index()
                .pivot(index="target_a_id", columns="target_b_id", values=sample)
            )
        except KeyError:
            # Not all samples are analyzed
            continue

        a, b, c, d = calibrate_sample_fitness(
            fitnesses.construct[sample],
            gp,
            fitnesses.smf_a[sample],
            parameters.loc[sample],
        )
        b = (
            b.reset_index()
            .melt(id_vars="target_a_id", value_name=sample)
            .set_index(["target_a_id", "target_b_id"])
        )

        d.name = sample

        cal = (a, b, c, d)

        for container, f in zip(calibrated, cal):
            container.append(f)

    calibrated = [pd.concat(container, axis=1) for container in calibrated]
    construct, genepair, smf_a, smf_b = calibrated

    calibrated = Fitnesses(construct, genepair, smf_a, smf_b)

    return calibrated


def calibrate_sample_fitness(construct, sq_genepair, smf_a, parameters):
    """Calibrate fitnesses

    The aim of calibrating fitness is to transform the measured fitness to the
    same quantities that are used for the regression to link it to the
    interaction scores

    Parameters
    ----------
    construct: pandas.DataFrame
    sq_genepair: pandas.DataFrame
        Gene A x Gene B fitness (in square format)
    smf_a : pandas.Series
        Gene A fitness
    parameters : pandas.DataFrame
        DataFrame with the slope ('m') and y-intercept ('b')
    """

    f_ntc_a = smf_a.loc["nontargeting"]

    calibrated_smf_b = parameters["b"] / parameters["a"]
    f_ntc_b = calibrated_smf_b.loc["nontargeting"]

    calibrated_genepair = (sq_genepair / parameters["a"].values) - f_ntc_a - f_ntc_b
    calibrated_smf_a = smf_a - f_ntc_a
    calibrated_smf_b = calibrated_smf_b - f_ntc_b

    scaled_construct = []
    for g, _df in construct.groupby("target_b_id"):
        m = parameters.loc[g, "a"]
        _df = (_df / m) - f_ntc_a - f_ntc_b

        scaled_construct.append(_df)

    scaled_construct = pd.concat(scaled_construct, axis=0)
    scaled_construct = scaled_construct.loc[construct.index]

    return scaled_construct, calibrated_genepair, calibrated_smf_a, calibrated_smf_b
