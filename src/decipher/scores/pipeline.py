"""Complete scoring pipeline"""

from collections import namedtuple

import numpy as np
import pandas as pd

from . import fitness
from . import utils
from . import interactions


Raw_scores = namedtuple(
    "Raw_scores",
    [
        "smf",
        "genepair_fitness",
        "regression_parameters",
        "interactions",
        "pivoted_index",
        "timepoint_names",
    ],
)


def calculate_interactions(
    counts,
    plasmid_col="plasmid",
    pseudocount=10,
    min_counts=None,
    geneA_name="target_a_id",
    geneB_name="target_b_id",
    ntc_name="nontargeting",
    experiment_name="experiment",
    force_symmetric=True,
):
    if min_counts is not None:
        counts = counts.astype(float)  # Make type consistent with nan
        counts[counts < min_counts] = np.nan

    genepair_fitness, resampled_genepair_f, end_cols = calculate_fitness(
        counts,
        plasmid_col,
        pseudocount,
        geneA_name=geneA_name,
        geneB_name=geneB_name,
        ntc_name=ntc_name,
    )

    pivoted_index = utils.pivot_multiindex(genepair_fitness.index)
    resampled_genepair_f = resampled_genepair_f[pivoted_index.values]

    smf_estimates = calculate_smf(resampled_genepair_f, n_samples=1000)

    resampled_raw_interactions, resampled_popts = (
        interactions.interactions_with_resampling(
            smf_estimates, resampled_genepair_f, n_resamples=1000
        )
    )

    raw_scores = Raw_scores(
        smf_estimates,
        resampled_genepair_f,
        resampled_popts,
        resampled_raw_interactions,
        pivoted_index,
        end_cols,
    )

    scores = interactions.process_raw_scores(
        raw_scores, experiment_name, force_symmetric=force_symmetric
    )
    # scores = utils.drop_symmetric_index(scores.set_index([geneA_name, geneB_name]))
    # scores = scores.set_index(['target_a_id', 'target_b_id'])

    return scores, raw_scores


def calculate_fitness(
    counts,
    start_col,
    pseudocount,
    geneA_name="target_a_id",
    geneB_name="target_b_id",
    ntc_name="nontargeting",
    resampling=True,
    make_fitness_symmetric=True,
):
    start_col, end_cols = utils.extract_columns(counts.columns, start_col=start_col)

    guidepair_fitness = fitness.log2enrichment_df(
        counts, start_col, end_cols, pseudocount=pseudocount, result_columns=end_cols
    )

    genepair_fitness = fitness.aggregate_guidepair(
        guidepair_fitness,
        geneA_name=geneA_name,
        geneB_name=geneB_name,
        agg=np.nanmedian,
    )

    if make_fitness_symmetric:
        guidepair_fitness = utils.make_symmetric(
            genepair_fitness, geneA_name, geneB_name, genepair_fitness.columns
        )

    ntc_fitness = genepair_fitness.loc[ntc_name, ntc_name].values

    if resampling:
        genepair_f, resampled_genepair_f = fitness.aggregate_guidepair_with_resampling(
            guidepair_fitness, end_cols
        )
        resampled_genepair_f -= ntc_fitness[..., np.newaxis]

        genepair_fitness -= ntc_fitness

        return genepair_f, resampled_genepair_f, end_cols

    genepair_fitness -= ntc_fitness

    return genepair_fitness, end_cols


def calculate_smf(resampled_genepair_fitness, n_samples=1000):
    selection_index = np.random.randint(
        0, n_samples, size=(*resampled_genepair_fitness.shape[:-1], n_samples)
    )
    sampled = np.take_along_axis(resampled_genepair_fitness, selection_index, axis=3)
    smf_estimates = np.median(sampled, axis=1)

    return smf_estimates
