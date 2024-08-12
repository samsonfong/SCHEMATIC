import numpy as np
from numpy.lib.function_base import percentile
import pandas as pd
from scipy.stats import percentileofscore

"""Common math utils"""


def standardize_column(df, column, center="mean"):
    """Standardize a column of a dataframe

    Paramters
    ---------
    df : pandas.DataFrame
        DataFrame that holds the column to standardize
    column : str or list of str
        Name(s) of the column to select
    center : str
        Either 'median' or 'mean'. This is the method used to define the center
        of the distribution

    Return
    ------
    df : pandas.DataFrame
        A copy of the dataframe with the column standardized
    """

    data = df[column].values

    if center == "median":
        center = np.nanmedian
    elif center == "mean":
        center = np.nanmean
    else:
        raise ValueError("Expected `center` to be either 'median' or 'mean'")

    df[column] = (data - center(data, axis=0)) / np.nanstd(data, axis=0)

    return df


def normalize(counts, axis=0):
    """Normalize along an axis"""

    total_counts = np.nansum(counts, axis=axis)

    return counts / total_counts


def quad_func(x, a, b, c):
    return a * x**2 + b * x + c


def linear_func(x, a, b):
    return a * x + b


def linear_func_factory(intercept):
    def inner(x, a):
        return a * x + intercept

    return inner


def split_dataframe_to_dicts(df, key_col, val_col, split_by, astype=float):
    dicts = {}
    for k, d in df.groupby(split_by):
        d = d.set_index(key_col)[val_col].astype(float).to_dict()
        dicts[k] = d

    return dicts


"""DataFrame utilities"""


def expand_collections(cols1, cols2):
    """Expand a collection by duplicating an object if needed

    Expect inputs to be either strings or iterables that contain __len__ method.
    If both inputs are not strings, check the inputs have the same lengths and
    return. If only one input is string, then force that input to have the
    same length as the other by duplicating the string in a list. If both
    inputs are strings, return both strings.
    """

    if isinstance(cols1, str) and isinstance(cols2, str):
        return cols1, cols2

    if isinstance(cols1, str) and not isinstance(cols2, str):
        cols1 = [cols1] * len(cols2)

    elif isinstance(cols2, str) and not isinstance(cols1, str):
        cols2 = [cols2] * len(cols1)

    if len(cols1) != len(cols2):
        raise ValueError("cols1 and col2 do not have the same columns!")

    return cols1, cols2


def validate_df_dtypes(df, expected_dtype, pure=False):
    """Validate that a dataframe is of the expected type"""

    dtypes = np.unique(df.dtypes)

    if pure:
        if len(dtypes) > 1:
            raise ValueError("DataFrame with multiple types found!")

    if not np.issubdtype(dtypes[0], expected_dtype):
        raise ValueError("Expected a DataFrame with numeric types!")


def extract_time(columns, sep="_"):
    return [int(i.split("_")[1][1:]) for i in columns]


def select(
    df, gA=None, gB=None, gA_col="target_a_id", gB_col="target_b_id", reverse=False
):
    """Subset the dataframe by two index"""

    gA_index = df[gA_col] == gA
    if gA is None:
        gA_index = True

    gB_index = df[gB_col] == gB
    if gB is None:
        gB_index = True

    subset = df.loc[gA_index & gB_index]
    if reverse:
        reverse_subset = select(
            df, gA=gB, gB=gA, gA_col=gA_col, gB_col=gB_col, reverse=False
        )
        subset = pd.concat([subset, reverse_subset], axis=0)

    return subset


def melt_upper_triu(df, k=0):
    keep = np.triu(np.ones(df.shape), k=k)
    keep = keep.astype("bool").reshape(df.size)

    return df.stack(dropna=False)[keep]


def extract_columns(columns, start_col="plasmid"):
    """Look for `start_col` in columns and separate it from the rest"""

    if start_col not in columns:
        raise ValueError(f"{start_col} was not found in the columns!")

    return start_col, [column for column in columns if column != start_col]


def pivot_multiindex(index):
    """Takes a 2-level index and pivot into a square index for pivoting"""
    df = pd.DataFrame(np.arange(len(index)), index=index).reset_index()
    if len(df.columns) != 3:
        raise ValueError("Expected index to have exactly 2 levels.")

    df = df.pivot(*df.columns)

    return df


def symmetric_average(x):
    """Makes x into a symmetric matrix by averaging

    If x has numbers in symmetric position, the two numbers are averages.
    If x only has only number in symmetric position, both positions will have
    that same number.
    If x has no numbers in either position, both positions will remain nan.
    """

    mask = ~np.isnan(x)
    sym = x.copy()
    sym = (sym + sym.T) / 2

    # cases where the sym pair is a number and nan
    single = mask & ~(mask & mask.T)
    sym[single] = x[single]
    sym[single.T] = x[single]

    return sym


def drop_symmetric_index(dataframe, drop_same=False):
    """Drops data where the reverse index is previously seen"""

    if len(dataframe.index.levels) != 2:
        raise ValueError("DataFrame needs to have exactly 2 levels of index.")

    seen = []
    index = []
    for ind, (i, j) in enumerate(dataframe.sort_index().index):
        if (i, j) in seen or (j, i) in seen:
            continue

        if drop_same and i == j:
            continue

        index.append(ind)
        seen.append((i, j))

    return dataframe.iloc[index]


def make_symmetric(data, index_col, columns_col, data_columns):
    """Make data symmetric"""

    if not np.all(np.isin([index_col, columns_col], data.columns)):
        data = data.reset_index()

    data = data.copy()
    overlap = np.intersect1d(data[index_col], data[columns_col])
    index = data[[index_col, columns_col]].isin(overlap).all(axis=1)

    for column in data_columns:
        square = data.loc[index].pivot(index_col, columns_col, column)
        square = symmetric_average(square)

        square = (
            square.reset_index()
            .melt(id_vars=index_col, value_name=column)
            .set_index([index_col, columns_col])
        )
        data.loc[index, column] = square.values

    return data


def get_background_values(scores, n_samples=1000, axis=1):
    backgrounds = []
    for _ in range(n_samples):
        values = scores.ravel().copy()
        np.random.shuffle(values)
        background = np.median(values.reshape(scores.shape), axis=axis)
        background = np.sort(background)
        backgrounds.append(background)

    backgrounds = np.stack(backgrounds, axis=-1)
    backgrounds = np.median(backgrounds, axis=-1)

    return backgrounds


def estimate_empirical_fdr(scores, background, side="left"):
    if side == "left":
        cdf = lambda x, i: percentileofscore(x, i)

    elif side == "right":
        cdf = lambda x, i: 100 - percentileofscore(x, i)

    else:
        raise ValueError("side can only be 'left' or 'right'")

    score_cdf = lambda i: cdf(scores, i)
    background_cdf = lambda i: cdf(background, i)

    fdr = []
    for i in scores:
        background_p = background_cdf(i)
        score_p = score_cdf(i)

        if score_p == 0:
            # For the minimum value, take the second value FDR
            # This should be a more conservative call for the minimum value
            if side == "left":
                score_p = score_cdf(np.sort(scores)[1])
            else:
                score_p = score_cdf(np.sort(scores)[-2])

        fdr.append(background_p / score_p)

    fdr = np.clip(fdr, 0, 1)

    return fdr


def square_array_to_flat_dataframes(square_array, index, columns, name=0):
    flat_index = [[i, c] for i in index for c in columns]
    df = pd.DataFrame(square_array.ravel(order="C"))
    df.index = pd.MultiIndex.from_tuples(flat_index)
    df.columns = [name]

    return df


def namedtuple_as_dict(named_tuple):
    package = {}
    for field, item in named_tuple._asdict().items():
        if isinstance(item, pd.DataFrame):
            package[f"{field}__index"] = item.index
            package[f"{field}__columns"] = item.columns
            package[f"{field}__values"] = item.values

        else:
            package[field] = item

    return package


def get_columns(columns, groups, reps=(1, 2)):
    columns = np.array(columns)

    if not isinstance(groups, dict):
        groups = {group: [group] for group in groups}

    cols = [i.split("_") for i in columns]

    indices = {}
    for name, group in groups.items():
        index = np.any(np.isin(cols, group), axis=1)
        index = columns[index]

        indices[name] = index

        if reps is not None:
            for rep in reps:
                _index = [ind for ind in index if f"_{rep}" in ind]
                indices[f"{name}_{rep}"] = _index

    return indices


def pop_controls(constructs):
    """Return the constructs whose targets have more than three guides

    These constructs should be for additional nontargeting controls.
    """

    index = []
    for _, row in constructs.reset_index().iterrows():
        a = int(row["probe_a_id"].split("-")[1])
        b = int(row["probe_b_id"].split("-")[1])

        if a > 3 or b > 3:
            index.append(False)
        else:
            index.append(True)

    index = np.array(index)

    controls = constructs.loc[~index]
    constructs = constructs.loc[index]

    return controls, constructs
