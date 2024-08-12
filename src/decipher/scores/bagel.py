"""Preparing results for BAGEL"""

import sys
import logging
import numpy as np
import pandas as pd
from decipher import PROJECT_BASE
from decipher.datasets.decipher_data import Decipher
from subprocess import Popen, PIPE
from pathlib import Path
from multiprocessing import Pool

# import logging

BASE = PROJECT_BASE / f"data/bagel"

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

fh = logging.FileHandler(BASE / "bagel.log")
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
log.addHandler(ch)

bagel_path = PROJECT_BASE / "bin/bagel"
essential_genes = pd.read_csv(bagel_path / "CEGv2.txt", sep="\t")["GENE"].values

decipher = Decipher()


def prepare_essentials(decipher, essential_genes, axis=None):
    if axis == 0:
        essential_pairs = np.intersect1d(decipher.geneA, essential_genes)

    elif axis == 1:
        essential_pairs = np.intersect1d(decipher.geneB, essential_genes)

    else:
        geneA_essentials = np.intersect1d(decipher.geneA, essential_genes)
        geneB_essentials = np.intersect1d(decipher.geneB, essential_genes)

        essential_pairs = [
            f"{geneA}__{geneB}"
            for geneA in geneA_essentials
            for geneB in geneB_essentials
        ]

        essential_pairs_A = [f"{geneA}__AAVS" for geneA in geneA_essentials]
        essential_pairs_B = [f"AAVS__{geneB}" for geneB in geneB_essentials]

        essential_pairs = essential_pairs_A + essential_pairs_B

    with open(input_files / "decipher-essential-pairs.txt", "w") as f:
        f.write("GENE\tHGNC_ID\nENTREZ_ID\n")
        f.write("\n".join(essential_pairs))


def prepare_nonessentials(axis=None):
    negative_as = ["AAVS", "nontargeting"]
    negative_as += [
        "PARP15",
        "TNKS",
        "SMO",
        "UNG",
        "PARP2",
    ]  # Top 5 negatives on Axis A according to DepMap

    negative_bs = ["AAVS", "nontargeting"]
    negative_bs += [
        "CDKN2B",
        "FBXW7",
        "MUTYH",
        "KDM5C",
        "SMAD4",
    ]  # Top 5 negatives on Axis B according to DepMap

    if axis == 0:
        negatives = negative_as
    elif axis == 1:
        negatives = negative_bs
    else:
        negatives = [
            f"{negative_a}__{negative_b}"
            for negative_a in negative_as
            for negative_b in negative_bs
        ]

    with open(input_files / "decipher-nonessential-pairs.txt", "w") as f:
        f.write("GENE\tHGNC_ID\tENTREZ_ID\n")
        for negative in negatives:
            f.write(f"{negative}\t\t\n")


# %%
def prepare_construct(construct_df, cell_line, filepath="fitness.txt", axis=None):
    if not isinstance(cell_line, list):
        cell_line = [cell_line]

    all_columns = []
    for cl in cell_line:
        inner_columns = []
        for col in construct_df.columns:
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

        all_columns += [last_r1, last_r2]

    data = construct_df[all_columns]
    names = prepare_index(data.index, axis=axis)
    data = data.assign(GENE=names)
    data = data.loc[~data.isnull().any(axis=1)].reset_index()

    data = data[["construct_id", "GENE", *all_columns]]
    data.columns = ["REAGENT_ID", "GENE", *all_columns]

    data.to_csv(filepath, index=False, sep="\t")

    return all_columns


def prepare_index(index, axis=None):
    index = index.to_frame().reset_index(drop=True)
    if axis is None:
        names = index["target_a_id"] + "__" + index["target_b_id"]
    elif axis == 0:
        names = index["target_a_id"]
    elif axis == 1:
        names = index["target_b_id"]
    else:
        raise ValueError

    return names.values


def run_bagel(name, cell_line, output_files, axis=None):
    """Run Bagel

    Paramters
    ---------
    axis : int or None
        If None, it will be treated as genepair. Axis = 0 means GeneA. Axis = 1
        means Gene B
    """

    log.info("Preparing construct fitness ")

    filepath = input_files / f"{name}_fc_for_bagel.csv"
    columns = prepare_construct(
        decipher.fitness.construct, cell_line, filepath=filepath, axis=axis
    )

    bagel_output_file = str(output_files / f"{name}_bagel_bf.tsv")

    args = [
        sys.executable,
        str(bagel_path / "BAGEL.py"),
        "bf",
        "-i",
        str(filepath),
        "-o",
        bagel_output_file,
        "-e",
        str(input_files / "decipher-essential-pairs.txt"),
        "-n",
        str(input_files / "decipher-nonessential-pairs.txt"),
        "-c",
        ",".join(columns),
    ]

    log.info("Running BAGEL bayes factor")
    log.info(" ".join(args))

    proc = Popen(args, stderr=PIPE, stdout=PIPE)
    stdout_data, stderr_data = proc.communicate()

    if stdout_data:
        log.info("Standard out")
        log.info(stdout_data.decode())

    if stderr_data:
        log.info("Standard error")
        log.info(stderr_data.decode())

    pr_filepath = str(output_files / f"{name}_bagel_pr.tsv")
    args = [
        sys.executable,
        str(bagel_path / "BAGEL.py"),
        "pr",
        "-i",
        bagel_output_file,
        "-o",
        pr_filepath,
        "-e",
        str(input_files / "decipher-essential-pairs.txt"),
        "-n",
        str(input_files / "decipher-nonessential-pairs.txt"),
    ]

    log.info("Running BAGEL precision recall")
    log.info(" ".join(args))

    proc = Popen(args, stderr=PIPE, stdout=PIPE)
    stdout_data, stderr_data = proc.communicate()

    if stdout_data:
        log.info("Standard out")
        log.info(stdout_data.decode())

    if stderr_data:
        log.info("Standard error")
        log.info(stderr_data.decode())


# %%
if __name__ == "__main__":
    run_map = [
        (0, "2.1-per-gene-A"),
        (1, "2.2-per-gene-B"),
        (None, "1-genepair"),
    ]

    for axis, run_name in run_map:
        # log.info(f"Runnning {run_name}...")
        base = BASE / run_name
        base.mkdir(exist_ok=True, parents=True)

        input_files = base / "input-files"
        output_files = base / "output-files"

        input_files.mkdir(exist_ok=True)
        output_files.mkdir(exist_ok=True)

        prepare_nonessentials(axis=axis)
        prepare_essentials(decipher, essential_genes, axis=axis)

        combo_map = decipher.combo_map.copy()
        combo_map.update({cell_line: [cell_line] for cell_line in decipher.cell_lines})

        kwargs = [(i, j, output_files, axis) for i, j in combo_map.items()]

        with Pool(16) as pool:
            pool.starmap(run_bagel, kwargs)
