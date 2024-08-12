"""Score Decipher"""

import logging
from pathlib import Path

import numpy as np

from decipher.scores import fitness, interactions, aggregate, cov
from decipher.datasets.depmap import DepMap

from decipher import PROJECT_BASE

DECIPHER_CELL_LINES = ["MCF7", "MDAMB231", "MCF10A", "CAL27", "CAL33", "A549", "A427"]

DECIPHER_CONTEXTS = {
    "brca": ["MCF7", "MDAMB231"],
    "hnsc": ["CAL27", "CAL33"],
    "nsclc": ["A549", "A427"],
    "pan-cancer": ["MCF7", "MDAMB231", "CAL27", "CAL33", "A549", "A427"],
    "KRAS": ["A427", "A549", "MDAMB231"],
    "POLQ": ["CAL27", "A549"],
    "PIK3CA": ["MCF7", "CAL33"],
    "NF2": ["MDAMB231", "CAL33"],
    "TP53": ["CAL27", "MDAMB231", "CAL33"],
    "TP53-wt": ["MCF10A", "MCF7", "A427", "A549"],
}

logger = logging.getLogger()


def score_decipher(counts_file):
    logger.info("Reading common essentials from the default path")
    # essential_genes = DepMap().load_common_essentials()
    essential_genes = DepMap().load_common_essentials(
        filename="Achilles_common_essentials.csv"
    )

    logger.info("Scoring fitness...")
    fitnesses = fitness.score_fitness(counts_file, essential_genes)

    logger.info("Scoring residuals...")
    residuals = interactions.score_residuals(
        fitnesses,
        pre_standardize=False,
        post_standardize=True,
    )

    logger.info("Calibrating fitness...")
    calibrated = fitness.calibrate_fitnesses(fitnesses, residuals.parameters)

    logger.info("Aggregating across cell lines...")
    cell_line_scores = aggregate.score_interactions(residuals, DECIPHER_CELL_LINES)

    logger.info("Aggregating across contexts...")
    combo_scores = aggregate.score_interactions(residuals, DECIPHER_CONTEXTS)

    logger.info("Calculating CoVs...")
    context_map = cov.get_context_sample_map(
        residuals.genepair_residuals.columns, DECIPHER_CELL_LINES, DECIPHER_CONTEXTS
    )
    covs = cov.determine_variations(
        residuals.guidepair_residuals,
        residuals.genepair_residuals.index,
        residuals.genepair_residuals.columns,
        context_map,
    )

    return fitnesses, residuals, calibrated, cell_line_scores, combo_scores, covs


def write_namedtuple(named_tuple, destination, prefix="", suffix=""):
    name = type(named_tuple).__name__
    out_dir = Path(destination) / name
    out_dir.mkdir(exist_ok=True, parents=True)

    for field, value in named_tuple._asdict().items():
        if hasattr(value, "to_csv"):
            value.to_csv(out_dir / f"{prefix}{field}{suffix}.csv")
        else:
            np.savez(out_dir / f"{prefix}{field}{suffix}.npz", value)


if __name__ == "__main__":
    counts_file = PROJECT_BASE / "data/decipher/decipher_counts.txt"
    counts_file = counts_file.resolve()
    logger.info(f"Reading counts file from {counts_file}")

    out_directory = PROJECT_BASE / "data/decipher/giscores-v9"
    out_directory = out_directory.resolve()
    logger.info(f"Writing results to {out_directory}")

    results = score_decipher(counts_file)

    logger.info("Serialing results...")

    prefixes = [
        "decipher_measured_",
        "decipher_",
        "decipher_calibrated_",
        "decipher_cell_lines_",
        "decipher_combos_",
        "decipher_cov_",
    ]
    for prefix, named_tuple in zip(prefixes, results):
        write_namedtuple(named_tuple, out_directory, prefix=prefix)
