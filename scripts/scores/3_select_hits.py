"""Select and classify interactions"""

import numpy as np
import pandas as pd
from pathlib import Path

from decipher import PROJECT_BASE
from decipher.datasets.decipher_data import Decipher


def define_essential_genes(bf_path, bf_cutoff, conditions):
    """Define which genes are essential

    Parameters
    ----------
    bf_cutoff: int or dict
        The bayes factor cutoff used to define what is essential. Can be a dict
        that maps condition to a cutoff.
    """

    if not isinstance(bf_cutoff, dict):
        bf_cutoff = {condition: bf_cutoff for condition in conditions}

    essential_genes_results = {}
    for condition in conditions:
        essential_genes_results[condition] = get_essential_genes_by_condition(
            condition, bf_path, bf_cutoff[condition]
        )

    essential_genes_df = []
    for name, genes in essential_genes_results.items():
        essential_genes_df.append(
            pd.DataFrame([genes], index=["genes"]).T.assign(condition=name)
        )

    essential_genes_df = pd.concat(essential_genes_df, axis=0)

    return essential_genes_df


def get_essential_genes_by_condition(condition, directory, bf_cutoff):
    directory_ab = Path(directory) / "1-genepair/output-files"
    gAB = pd.read_csv(directory_ab / f"{condition}_bagel_bf.tsv", sep="\t")
    gA, gB = extract_single_genes_from_bagel(gAB)

    decipher_essentials = np.union1d(
        gA.loc[gA["BF"] > bf_cutoff, "geneA"].values,
        gB.loc[gB["BF"] > bf_cutoff, "geneB"].values,
    )

    return decipher_essentials


def extract_single_genes_from_bagel(bagel_df, ntc="nontargeting"):
    names = np.array([i.split("__") for i in bagel_df["GENE"]])
    bagel_df = bagel_df.assign(geneA=names[:, 0], geneB=names[:, 1])

    gA = bagel_df.loc[bagel_df["geneB"] == ntc]
    gB = bagel_df.loc[bagel_df["geneA"] == ntc]

    return gA, gB


def annotate_unique_essentials(
    essentials, condition="condition", conserved="pan-cancer", genes="genes"
):
    pan_genes = essentials.loc[essentials[condition] == conserved, genes].values

    annotated = []
    for condition, df in essentials.groupby(condition):
        in_conserved = [i in pan_genes for i in df[genes]]
        annotated.append(df.assign(in_conserved=in_conserved))

    annotated = pd.concat(annotated, axis=0)

    return annotated


def select_all_synthetic_lethalities(
    bf_path,
    essential_genes_results,
    conditions,
    decipher_scores,
    covs,
):
    all_hits = []
    for condition in conditions:
        decipher_genepair_bagel = pd.read_csv(
            bf_path / f"1-genepair/output-files/{condition}_bagel_bf.tsv",
            sep="\t",
            index_col=[0],
        )
        decipher_genepair_bagel.index = pd.MultiIndex.from_tuples(
            [tuple(i.split("__")) for i in decipher_genepair_bagel.index]
        )

        conserved = condition == "pan-cancer"

        sl_selection = {
            "comfortable": dict(
                effect_size=10,
                fdr=0.1,
                bagel_cutoff=5,
                conserved=conserved,
                no_essentials=False,
            ),
        }

        for name, sl_kwargs in sl_selection.items():
            essential_genes_set = (
                (
                    essential_genes_results.loc[
                        essential_genes_results["condition"] == condition, "genes"
                    ].values,
                    "",
                ),
            )

            for essential_genes, _name in essential_genes_set:
                hits = select_synthetic_lethalities(
                    decipher_scores,
                    condition,
                    decipher_genepair_bagel,
                    covs,
                    essential_genes,
                    **sl_kwargs,
                )

                hits = pd.DataFrame(hits, columns=["geneA", "geneB"])
                hits = hits.assign(
                    a_is_essential=hits["geneA"].isin(essential_genes),
                    b_is_essential=hits["geneB"].isin(essential_genes),
                    condition=f"{condition}{_name}",
                    method=name,
                )

                all_hits.append(hits)

    all_hits = pd.concat(all_hits, axis=0)

    return all_hits


def select_synthetic_lethalities(
    decipher_scores,
    condition,
    genepair_bagel,
    covs,
    essential_genes,
    effect_size=5,
    fdr=0.1,
    bagel_cutoff=5,
    conserved=True,
    no_essentials=True,
):
    neg = decipher_scores[condition] < 0
    hits = decipher.filter(effect_size, fdr=fdr)[condition]

    index = neg & hits

    if bagel_cutoff is not None:
        gps_are_essential = genepair_bagel["BF"] > bagel_cutoff
        gps_are_essential.index.names = ["target_a_id", "target_b_id"]

        index = index & gps_are_essential

    if conserved:
        conserved_ = covs[condition.replace("-", "_")] < 5.81
        index = index & conserved_

    pairs_of_interest = decipher_scores.loc[index, condition]

    if no_essentials:
        pairs_of_interest = pairs_of_interest.loc[
            ~pairs_of_interest.reset_index()[["target_a_id", "target_b_id"]]
            .isin(essential_genes)
            .any(axis=1)
            .values
        ]

    poi_flat = [(i, j) for i, j in pairs_of_interest.index]

    return poi_flat


def annotate_unique_pairs(
    pair_hits,
    method="method",
    condition="condition",
    conserved="pan-cancer",
    genes=["geneA", "geneB"],
):
    all_hits_annotated = []
    for _, _df in pair_hits.groupby(method):
        pancan = _df.loc[_df[condition] == conserved, genes].values
        pancan = [(i, j) for i, j in pancan]

        for _, df in _df.groupby(condition):
            hits = df[genes].values
            in_conserved = np.array([(i, j) in pancan for i, j in hits])
            df = df.assign(in_pancan=in_conserved)

            all_hits_annotated.append(df)

    all_hits_annotated = pd.concat(all_hits_annotated, axis=0)

    return all_hits_annotated


def to_set_of_sets(values):
    set_of_sets = []
    for value in values:
        set_of_sets.append(frozenset(value))

    return set(set_of_sets)


def categorize_hit(hits):
    tissues = list(decipher.combo_map.keys())
    cell_lines = ["A549", "A427", "CAL33", "CAL27", "MDAMB231", "MCF7", "MCF10A"]

    if hits["pan-cancer"]:
        return "pan-cancer"

    tissue = [t for t in tissues if hits[t]]

    if tissue:
        if len(tissue) == 1:
            return tissue[0]
        else:
            return "multi"

    cell_line = [t for t in cell_lines if hits[t]]

    if cell_line:
        if len(cell_line) == 1:
            return cell_line[0]
        else:
            return "multi"

    return None


if __name__ == "__main__":
    decipher = Decipher()

    output = PROJECT_BASE / "data/decipher/functions"
    scores_dir = PROJECT_BASE / "data/decipher/giscores-v9"
    hits_dir = PROJECT_BASE / "data/decipher/hits"
    hits_dir.mkdir(exist_ok=True)

    bf_path = PROJECT_BASE / "data/bagel"

    decipher = Decipher()

    conditions = [
        "pan-cancer",
        "MDAMB231",
        "MCF7",
        "MCF10A",
        "CAL33",
        "CAL27",
        "A549",
        "A427",
        "brca",
        "hnsc",
        "nsclc",
        "KRAS",
        "POLQ",
        "PIK3CA",
        "NF2",
        "TP53",
        "TP53-wt",
    ]

    essential_gene_results = define_essential_genes(
        bf_path=bf_path,
        bf_cutoff={condition: 5 for condition in conditions},
        conditions=conditions,
    )
    essential_gene_results = annotate_unique_essentials(essential_gene_results)

    essential_gene_results.to_csv(
        hits_dir / "decipher_essential_genes.csv", index=False
    )

    covs = pd.read_csv(
        scores_dir / "cov/decipher_cov_cov.csv",
        index_col=[0, 1],
    )

    all_hits = select_all_synthetic_lethalities(
        bf_path,
        essential_gene_results,
        conditions=conditions,
        decipher_scores=decipher.scores,
        covs=covs,
    )
    all_hits = annotate_unique_pairs(all_hits)

    # Remove negative controls, same-same, and AB-BA
    controls = ["AAVS", "nontargeting"]
    no_controls = ~all_hits[["geneA", "geneB"]].isin(controls).any(axis=1)
    no_same = all_hits["geneA"] != all_hits["geneB"]

    seen = []
    skip = []
    for _, row in all_hits.iterrows():
        record = (row["geneA"], row["geneB"], row["condition"], row["method"])
        flip = (row["geneB"], row["geneA"], row["condition"], row["method"])

        skip.append(flip in seen)
        seen.append(record)

    no_flip = ~np.array(skip)
    all_hits = all_hits.loc[no_controls & no_same & no_flip]

    all_hits.to_csv(hits_dir / "decipher_synthetic_essential_genes.csv")

    namemap = {
        "pan-cancer": "Pan-cancer",
        "brca": "Breast",
        "hnsc": "Oropharyngeal",
        "nsclc": "Lung",
        "multi": "Multiple contexts",
    }

    hits = all_hits.loc[all_hits["method"] == "comfortable"].assign(value=True)
    hits = hits.loc[hits["condition"].isin(conditions)]
    hits = hits.pivot(
        index=["geneA", "geneB"], columns="condition", values="value"
    ).fillna(False)

    # Account for A-B and B-A
    categories = {}
    seen = []
    for index, row in hits.iterrows():
        index = frozenset(index)

        if index in seen:
            _index = tuple(index)
            tmp = hits.loc[[_index, _index[::-1]]]
            row = tmp.any(axis=0)

        categories[index] = categorize_hit(row)
        seen.append(index)

    categories = pd.Series(categories)

    categories_, counts = np.unique(categories, return_counts=True)
    counts = pd.Series(counts, index=categories_)
    counts = pd.DataFrame(counts).assign(fraction=counts / counts.sum())
    counts = counts.loc[conditions]

    counts.index = [namemap.get(i, i) for i in counts.index]

    category_namemap = {
        k: k for k in list(decipher.combo_map.keys()) + list(decipher.cell_lines)
    }
    category_namemap.update(
        {
            "pan-cancer": "Pan-cancer",
            "multi": "Several contexts",
            "hnsc": "Oropharyngeal",
            "brca": "Breast",
            "nsclc": "Lung",
        }
    )

    genepairs = pd.DataFrame(
        [[i, j] for i, j in categories.index], columns=["Gene A", "Gene B"]
    )
    synthtetic_essential_genes_table = genepairs.assign(context=categories.values)
    synthtetic_essential_genes_table["context"] = synthtetic_essential_genes_table[
        "context"
    ].map(category_namemap)

    synthtetic_essential_genes_table.to_csv(
        hits_dir / "synthtetic_essential_genes_table.csv"
    )
