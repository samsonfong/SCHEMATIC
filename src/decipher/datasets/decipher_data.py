"""Retrieve decipher data"""

from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import auc, roc_curve

from ..scores import interactions
from ..scores.fitness import Fitnesses

Scores = namedtuple("Scores", ["scores", "fdrs", "cutoffs", "groups"])

from .. import PROJECT_BASE

score_path = Path(PROJECT_BASE / "data/decipher/giscores-v9/")


class Decipher(object):
    """Retrieve all decipher data and instantiate filter sets"""

    combo_map = {
        "brca": ["MDAMB231", "MCF7"],
        "hnsc": ["CAL27", "CAL33"],
        "nsclc": ["A549", "A427"],
        "pan-cancer": ["MDAMB231", "MCF7", "CAL27", "CAL33", "A549", "A427"],
        "KRAS": ["A427", "A549", "MDAMB231"],
        "POLQ": ["CAL27", "A549"],
        "PIK3CA": ["MCF7", "CAL33"],
        "NF2": ["MDAMB231", "CAL33"],
        "TP53": ["CAL27", "MDAMB231", "CAL33"],
        "TP53-wt": ["MCF10A", "MCF7", "A427", "A549"],
    }

    cell_lines = ["A549", "A427", "CAL27", "CAL33", "MDAMB231", "MCF7", "MCF10A"]

    control = "AAVS"

    def __init__(self, score_path=score_path):
        self.score_path = Path(score_path)

        self.cell_line_scores = read_scores(
            score_path=self.score_path, scores_type="cell_lines"
        )
        self.tissue_scores = read_scores(
            score_path=self.score_path, scores_type="combos"
        )
        self.scores = pd.concat(
            [
                self.cell_line_scores.scores[self.cell_lines],
                self.tissue_scores.scores[list(self.combo_map.keys())],
            ],
            axis=1,
        )
        self.fdrs = pd.concat(
            [
                self.cell_line_scores.fdrs[self.cell_lines],
                self.tissue_scores.fdrs[list(self.combo_map.keys())],
            ],
            axis=1,
        )

        self.fitness = read_fitness(score_path=self.score_path, which="calibrated")
        self.measured_fitness = read_fitness(
            score_path=self.score_path, which="measured"
        )

        self._init_filters()

        # Get genes
        self.genepairs = self.fitness.genepair.reset_index()[
            ["target_a_id", "target_b_id"]
        ]
        self.geneA = self.genepairs["target_a_id"].unique()
        self.geneB = self.genepairs["target_b_id"].unique()
        self.genes = np.union1d(self.geneA, self.geneB)

        # Regression parameters
        self.parameters = pd.read_csv(
            self.score_path / "Residuals/decipher_parameters.csv", index_col=[0, 1]
        )
        self.scalers = pd.read_csv(
            self.score_path / "Residuals/decipher_scalers.csv", index_col=[0, 1]
        )
        self.genepair_residuals = pd.read_csv(
            self.score_path / "Residuals/decipher_genepair_residuals.csv",
            index_col=[0, 1],
        )

        self.cov = pd.read_csv(
            self.score_path / "cov/decipher_cov_cov.csv", index_col=[0, 1]
        )

    def _init_filters(self):
        self.score_filter = Filters(
            pd.concat(
                [self.cell_line_scores.scores, self.tissue_scores.scores], axis=1
            ),
            pd.concat([self.cell_line_scores.fdrs, self.tissue_scores.fdrs], axis=1),
        )

        self.min_filter = Filters(
            pd.concat([self.cell_line_min_scores, self.tissue_min_scores], axis=1),
        )

        self.max_filter = Filters(
            pd.concat([self.cell_line_max_scores, self.tissue_max_scores], axis=1),
        )

    def filter(
        self,
        ci=5,
        fdr=0.3,
        with_fitness=False,
        fitness_ci=5,
        return_scores=False,
        cell_lines=None,
    ):
        """Filter scores based on effect size, fdr, and fitness

        Paramters
        ---------
        ci : float
            Confidence interval for the *effect size*. It's defined as (100 - the
            interval used for the null distribution). For instance, if ci = 5,
            the hits has to be outside the 95% confidence interval of the AAVS.
            Size of the residual and requires that the residual is outside the
            `ci` confidence interval of the negative control, which is by
            default defined as all interactions that include AAVS
        fdr : float
            The FDR cutoff to use to filter score
        with_fitness : bool
            If True, the paired fitness will be compared to the single.
        fitness_ci : float
            The confidence interval for the interaction scores defined by the
            minimum or maximum definition. Being outside of this interval
            guarantees that the double will be more deleterious that the individual
            (for negative interactions)
        return_scores : bool
            If True, the scores passing the desired filter will be return.
            Otherwise, a DataFrame of True or False will be return. True refers
            to genepairs that pass filter.
        cell_lines : None or str or list of str
            If None, all cell lines and combos will be returned. If only a string,
            it's expected that it is the name of the individual cell line. If it
            is a collection, it will be used to index.

        Returns
        -------
        passing : pandas.DataFrame
            See `return_scores` for more details.
        """

        left_effect = ci / 2
        right_effect = 100 - left_effect

        passing = (
            self.score_filter.on_left_effect_size(left_effect)
            | self.score_filter.on_right_effect_size(right_effect)
        ) & self.score_filter.on_fdr(fdr)

        if with_fitness:
            min_passing = (self.scores <= 0) & self.min_filter.on_left_effect_size(
                fitness_ci / 2
            )
            max_passing = (self.scores > 0) & self.max_filter.on_right_effect_size(
                100 - fitness_ci / 2
            )

            fit_passing = min_passing | max_passing

            passing = passing & fit_passing

        if cell_lines is None:
            columns = self.cell_lines + list(self.combo_map.keys())
        elif isinstance(cell_lines, str):
            columns = [cell_lines]
        else:
            columns = cell_lines

        passing = passing[columns]

        if return_scores:
            results = []
            for column in passing.columns:
                result = self.scores.loc[passing[column], [column]]
                result.columns = ["scores"]
                result = result.assign(cell_line=column)

                results.append(result)

            results = pd.concat(results, axis=0)

            return results

        return passing

    def get_final_timepoints(self):
        """Get the final timepoints for each cell line"""

        all_columns = {}
        for cl in self.cell_lines:
            inner_columns = []
            for col in self.fitness.construct.columns:
                if cl in col:
                    inner_columns.append(col)

            # Getting the last timepoint in each replicate
            get_t = lambda x: int(x.split("_")[1][1:])
            last_r1 = sorted(
                [column for column in inner_columns if "_1" in column], key=get_t
            )[-1]
            last_r2 = sorted(
                [column for column in inner_columns if "_2" in column], key=get_t
            )[-1]

            all_columns[cl] = [last_r1, last_r2]

        return all_columns


def read_scores(score_path=score_path, scores_type="cell_lines"):
    cl_scores = pd.read_csv(
        score_path / "Scores" / f"decipher_{scores_type}_scores.csv", index_col=[0, 1]
    )
    cl_fdr = pd.read_csv(
        score_path / "Scores" / f"decipher_{scores_type}_fdrs.csv", index_col=[0, 1]
    )
    cl_cutoffs = pd.read_csv(
        score_path / "Scores" / f"decipher_{scores_type}_cutoffs.csv", index_col=0
    )
    cell_lines = [i for i in cl_scores.columns if "_" not in i]

    scores = Scores(cl_scores, cl_fdr, cl_cutoffs, cell_lines)

    return scores


def read_fitness(score_path=score_path, which="calibrated"):
    filepath = lambda x: score_path / "Fitnesses" / f"decipher_{which}_{x}.csv"

    construct = pd.read_csv(filepath("construct"), index_col=[0, 1, 2, 3, 4])
    genepair = pd.read_csv(filepath("genepair"), index_col=[0, 1])
    smf_a = pd.read_csv(filepath("smf_a"), index_col=[0])
    smf_b = pd.read_csv(filepath("smf_b"), index_col=[0])

    fitness = Fitnesses(construct, genepair, smf_a, smf_b)

    return fitness


class Filters(object):
    def __init__(self, scores, fdr=None):
        self.scores = scores
        self.fdr = fdr

    def on_scores(self, cutoff):
        return self.scores < cutoff

    def on_effect_size(self, neg_pct):
        # Keeping this for legacy code
        negative = self._get_negative_scores()
        cutoff = np.percentile(negative, neg_pct, axis=0)

        return self.scores < cutoff

    def on_left_effect_size(self, neg_pct):
        return self.on_effect_size(neg_pct)

    def on_right_effect_size(self, pos_pct):
        negative = self._get_negative_scores()
        cutoff = np.percentile(negative, pos_pct, axis=0)

        return self.scores > cutoff

    def on_fdr(self, fdr):
        return self.fdr < fdr

    def _get_negative_scores(self, negative="AAVS"):
        negative_df = pd.concat(
            [self.scores.xs(negative, level=0), self.scores.xs(negative, level=1)],
            axis=0,
        )

        return negative_df


def essentiality_threshold(fitness_data, essential_index):
    """Establish the fitness threshold to call an essential gene

    Columns are expected to be samples while rows are genes or genepairs
    """

    # essential_index = fitness_data.index.isin(essential_genes)
    non_essential_index = ~essential_index  # TODO: Not sure this is a good idea

    boundaries = []
    for column in fitness_data.columns:
        essential_f = fitness_data.loc[essential_index, column]
        non_essential_f = fitness_data.loc[non_essential_index, column]

        values = np.hstack(
            [-1 * non_essential_f, -1 * essential_f]
        )  # roc_curve expect higher values for positive class
        classes = np.hstack(
            [np.zeros(non_essential_f.shape), np.ones(essential_f.shape)]
        )

        boundary = find_elbow_from_roc(values, classes)
        boundaries.append(boundary)

    boundaries = -1 * pd.Series(boundaries, index=fitness_data.columns)

    return boundaries


def find_elbow_from_roc(values, classes):
    # Determining threhsold
    fpr, tpr, thresholds = roc_curve(classes, values, pos_label=1)
    auroc = auc(fpr, tpr)
    index = find_elbow(
        np.vstack([tpr, fpr]).T
    )  # Need to be reversed because the function is finding the bottom curvature

    threshold = thresholds[index]

    return threshold


def find_elbow(data):
    """Source
    https://datascience.stackexchange.com/questions/57122/in-elbow-curve-how-to-find-the-point-from-where-the-curve-starts-to-rise
    """
    theta = get_data_radiant(data)

    # make rotation matrix
    co = np.cos(theta)
    si = np.sin(theta)
    rotation_matrix = np.array(((co, -si), (si, co)))

    # rotate data vector
    rotated_vector = data.dot(rotation_matrix)

    # return index of elbow
    return np.where(rotated_vector == rotated_vector.min())[0][0]


def get_data_radiant(data):
    return np.arctan2(
        data[:, 1].max() - data[:, 1].min(), data[:, 0].max() - data[:, 0].min()
    )


def get_decipher_hits(hits_file):
    if hits_file is None:
        hits_file = (
            PROJECT_BASE / "data/decipher/decipher_synthetic_essential_categorized.csv"
        )

    decipher_hits = pd.read_csv(hits_file)

    return decipher_hits
