"""Module for GDSC data"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from .. import PROJECT_BASE

DATA_PATH = PROJECT_BASE / "data/gdsc"

class GDSC(object):
    """GDSC data"""

    def __init__(self, datapath=DATA_PATH):
        self.datapath = datapath

    def load_all(self):
        self.load_mutations()
        self.load_robust_expression()
        self.load_drug_response()
        self.load_sample_info()

        return self

    def load_mutations(self):
        self.mutations = pd.read_csv(DATA_PATH / "mutations_all_20220510.csv")
        # mutations = mutations.loc[mutations["effect"] != "silent"] # this was too
        # relaxed
        self.mutations = self.mutations.loc[
            self.mutations["protein_mutation"] != "-"
        ]  # this might be too stringent
        self.sq_mutations = (
            self.mutations[["model_name", "gene_symbol"]]
            .drop_duplicates()
            .assign(_mutated=1)
            .pivot(index="model_name", columns="gene_symbol", values="_mutated")
            .fillna(0)
            .astype(bool)
        )

    def load_robust_expression(self):
        self.copy_number = pd.read_csv(
            DATA_PATH
            / "WES_pureCN_CNV_genes_20221213/WES_pureCN_CNV_genes_total_copy_number_20221213.csv",
            skiprows=[1, 2, 3],
            index_col=0,
            low_memory=False,
        )
        self.expression = (
            pd.read_csv(
                DATA_PATH / "rnaseq_all_20220624/rnaseq_fpkm_20220624.csv",
                skiprows=[0, 2, 3, 4],
            )
            .drop(columns=["model_name"])
            .groupby("Unnamed: 1")
            .median()
        )

        index = self.copy_number.index.intersection(self.expression.index)
        columns = self.copy_number.columns.intersection(self.expression.columns)

        copy_number = self.copy_number.loc[index, columns]
        expression = self.expression.loc[index, columns]
        underexpression = expression < np.percentile(expression, 25, axis=1).reshape(
            -1, 1
        )

        self.robust_cn = (copy_number < 2) & underexpression
        self.robust_cn = self.robust_cn.T

    def load_drug_response(self):
        # Separate by comma followed by a non-whitespace character
        self.drug1 = pd.read_csv(
            self.datapath / "gdsc1_drug_data.csv", sep=",(?=\S)", engine="python"
        )
        self.drug2 = pd.read_csv(
            self.datapath / "gdsc2_drug_data.csv", sep=",(?=\S)", engine="python"
        )

        self.drug_data = {"GDSC1": self.drug1, "GDSC2": self.drug2}

        columns = ["Drug name", "Cell line name", "IC50"]
        for k, v in self.drug_data.items():
            self.drug_data[k] = (
                v[columns]
                .groupby(["Drug name", "Cell line name"])
                .median()
                .reset_index()
                .pivot(index="Drug name", columns="Cell line name", values="IC50")
            )

    def load_sample_info(self):
        self.drug_info = pd.read_csv(
            DATA_PATH / "drug-list.csv", header=None, skiprows=[0]
        )
        self.drug_info.columns = [
            "drug_id",
            "name",
            "synonyms",
            "targets",
            "target_pathways",
            "pubchem",
            "dataset",
            "sample size",
            "count",
            None,
        ]

        self.cell_line_info = pd.read_csv(DATA_PATH / "cell-line-info.txt", sep="\t")
        self.cell_line_info["tissue"] = self.cell_line_info["Source Name"].apply(
            lambda x: x.split("_")[-2].lower()
        )

    def get_cell_lines_from_tissues(self, tissues):
        """Get the name of the cell lines of a particular tissue"""

        return self.cell_line_info.loc[
            self.cell_line_info["tissue"].isin(tissues), "Characteristics[cell line]"
        ].values

    def select_drug_data_by_drugname(self, drug_name, dataset="GDSC1"):
        drug_data = self.drug_data[dataset]
        drug_data = drug_data.loc[drug_name]

        return drug_data

    def separate_cell_lines(self, mutations, mutation_data_name="sq_mutations"):
        mutation_data = getattr(self, mutation_data_name)
        mutation_data = mutation_data[mutations]

        if isinstance(mutation_data, pd.DataFrame):
            mutation_data = mutation_data.any(axis=1)

        mutated_lines, wt_lines = separate_index(mutation_data)

        return mutated_lines, wt_lines

    def calculate_sensitivity(
        self,
        drug_name,
        mutations,
        mutation_data_name="sq_mutations",
        dataset="GDSC1",
        min_samples=3,
        lineage=None,
    ):
        mutated_lines, wt_lines = self.separate_cell_lines(
            mutations, mutation_data_name=mutation_data_name
        )

        drug_data = self.select_drug_data_by_drugname(drug_name, dataset=dataset)

        cell_lines = drug_data.index
        if lineage is not None:
            self.filter_cell_lines_by_tissue(cell_lines, lineage)

        mutated_lines = np.intersect1d(mutated_lines, cell_lines)
        wt_lines = np.intersect1d(wt_lines, cell_lines)

        mutated_auc = drug_data[mutated_lines].dropna().values
        wt_auc = drug_data[wt_lines].dropna().values

        if (len(mutated_auc) < min_samples) & (len(wt_auc) < min_samples):
            return

        diff = np.median(mutated_auc) - np.median(wt_auc)
        stat, pval = ttest_ind(mutated_auc, wt_auc, alternative="less")

        return diff, stat, pval

    @property
    def target_drugs_map(self):
        # TODO: Some of the mapping isn't quite right (PARP5 for instance)

        if hasattr(self, "_target_drugs_map"):
            return self._target_drugs_map

        _target_drugs_map = defaultdict(list)
        for drug_name, targets in self.drug_info[["name", "targets"]].values:
            try:
                targets = targets.split(",")
            except AttributeError:
                continue

            for target in targets:
                _target_drugs_map[target.strip()].append(drug_name)

        self._target_drugs_map = {k: list(set(v)) for k, v in _target_drugs_map.items()}

        return self._target_drugs_map

    @property
    def drug_dataset_map(self):
        if hasattr(self, "_drug_dataset_map"):
            return self._drug_dataset_map

        self._drug_dataset_map = (
            self.drug_info.set_index("name")["dataset"]
            .reset_index()
            .groupby("name")
            .apply(lambda x: list(x["dataset"]))
        )

        return self._drug_dataset_map

    def filter_cell_lines_by_tissue(self, cell_lines, lineage):
        if isinstance(lineage, str):
            lineage = [lineage]

        cell_lines_in_lineage = self.cell_line_info.loc[
            self.cell_line_info["tissue"].isin(lineage), "Characteristics[cell line]"
        ].values

        cell_lines_in_lineage = np.intersect1d(cell_lines, cell_lines_in_lineage)

        return cell_lines_in_lineage


def separate_index(series):
    true_index = series.index[series.values]
    false_index = series.index[~series.values]

    return true_index, false_index


if __name__ == "__main__":
    gdsc = GDSC().load_all()
