"""TCGA related code"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, "D:/work-vault/2-Projects/decipher/decipher-dvc")
import os

from decipher_v5.datasets.decipher_data import Decipher

os.environ["DATA_PATH"] = "D:/data"

from decipher.datasets.hierarchy import Hierarchy
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import fisher_exact


def decipher_sl_in_tcga(genepairs, tcga_map):
    results = []

    def validate_with(tcga):
        survival_times, pvalue, lengths = validate_genepair(
            tcga, geneA, geneB, "mutations"
        )
        results.append(
            [geneA, geneB, tcga.name, "mutations", *survival_times, pvalue, *lengths]
        )

        survival_times, pvalue, lengths = validate_genepair(
            tcga, geneA, geneB, "filtered_underexpression"
        )
        results.append(
            [
                geneA,
                geneB,
                tcga.name,
                "filtered_underexpression",
                *survival_times,
                pvalue,
                *lengths,
            ]
        )

    for _, row in genepairs.iterrows():
        geneA, geneB, context = row["geneA"], row["geneB"], row["context"]
        tcga = tcga_map[context]
        if isinstance(tcga, list):
            for tcga_ in tcga:
                validate_with(tcga_)

        else:
            validate_with(tcga)

    return results


def decipher_sl_across_in_tcga(sl_across, tcga_map, membership_map):
    results = []

    def validate_with(tcga):
        survival_times, pvalue, lengths = validate_genepair(
            tcga, system_genes, gene, "mutations", systems=True
        )
        results.append(
            [
                system_name,
                gene,
                tcga.name,
                "mutations",
                *survival_times,
                pvalue,
                *lengths,
            ]
        )

        survival_times, pvalue, lengths = validate_genepair(
            tcga, system_genes, gene, "filtered_underexpression", systems=True
        )
        results.append(
            [
                system_name,
                gene,
                tcga.name,
                "filtered_underexpression",
                *survival_times,
                pvalue,
                *lengths,
            ]
        )

    for _, row in sl_across.iterrows():
        system_name, gene, context = row["name"], row["induced_by"], row["condition"]
        system_genes = membership_map[system_name]

        tcga = tcga_map[context]
        if isinstance(tcga, list):
            for tcga_ in tcga:
                validate_with(tcga_)

        else:
            validate_with(tcga)

    return results


def validate_genepair(tcga, geneA, geneB, data_type, systems=False):
    if systems:
        outputs = tcga.select_groups_for_sl_across(data_type, geneA, geneB)
    else:
        outputs = tcga.select_groups(data_type, geneA, geneB)

    not_both = outputs[0].union(outputs[1]).union(outputs[2])
    both = outputs[3]

    not_both_len = len(not_both)
    both_len = len(both)

    if (not_both_len < 3) or (both_len < 3):
        return (None, None), None, (not_both_len, both_len)

    _, not_both_km = tcga.kaplan_meier(sample_ids=not_both, plot=False)
    not_both_mst = not_both_km.median_survival_time_

    _, both_km = tcga.kaplan_meier(sample_ids=both, plot=False)
    both_mst = both_km.median_survival_time_

    logrank_pval = tcga.logrank_test(not_both, both).p_value

    return (not_both_mst, both_mst), logrank_pval, (not_both_len, both_len)

    # output_lengths = [len(output) for output in outputs]
    # empty_outputs = [output_length <= 3 for output_length in output_lengths]
    # if any(empty_outputs):
    #     return [None] * 4, [None] * 3, output_lengths

    # median_survival_times = []
    # p_values = []
    # for ind, output in enumerate(outputs):
    #     _, km = tcga.kaplan_meier(sample_ids=output, plot=False)
    #     median_survival_times.append(km.median_survival_time_)

    #     if ind == 3:
    #         continue

    #     p_values.append(tcga.logrank_test(output, outputs[-1]).p_value)

    # return median_survival_times, p_values, output_lengths


def filter_nest(decipher):
    """Getting the main decipher_nest"""

    filter_kwargs = dict(min_fraction=0.1, min_overlap=3, min_size=5)

    nest = Hierarchy.from_ndex("9a8f5326-aa6e-11ea-aaef-0ac135e8bacf")
    decipher_nest, selected_decipher_genes = nest.subgraph(
        decipher.genes, "NEST", **filter_kwargs
    )

    return nest, decipher_nest, selected_decipher_genes


def get_sl_across():
    tissue_map = {
        "brca": "brca",
        "hnsc": "hnsc",
        "nsclc": "luad",
        "pan-cancer": None,
        "MDAMB231": "brca",
        "MCF7": "brca",
        "MCF10A": "brca",
        "CAL27": "hnsc",
        "CAL33": "hnsc",
        "A549": "luad",
        "A427": "luad",
    }

    sl_across_file = "D:/work-vault/2-Projects/decipher/decipher-dvc/data/2_scores/functions-221115/NEST/functions-by-synthetic-essential-gene-partners.csv"
    sl_across = pd.read_csv(sl_across_file)
    sl_across = sl_across.loc[
        (sl_across["fdr"] < 0.3)
        & (sl_across["ess-binom-fdr"] < 0.3)
        & (sl_across["stringency"] == "lenient")
        & (sl_across["subset"] == "complete")
        & (sl_across["context"] == "all")
        & (sl_across["condition"].isin(list(tissue_map.keys())))
    ]
    sl_across = sl_across.assign(tcga_cohort=sl_across["condition"].map(tissue_map))

    return sl_across


class ClinicalData(object):
    """Getting TCGA data"""

    def __init__(self, raw_mutations, expression, cna, patients, samples):
        self.raw_mutations = raw_mutations
        self.expression = expression
        self.cna = cna
        self.patients = patients
        self.samples = samples

        self.patient_sample_map, self.sample_patient_map = self.map_patient_samples(
            self.samples
        )

        # deleterious_mutations = self.remove_benign_mutations(self.raw_mutations)
        # self.mutations = self.pivot_mutations_table(deleterious_mutations)
        self.mutations = self.pivot_mutations_table(self.raw_mutations)

    @classmethod
    def from_cbioportal_directory(cls, data_directory):
        data_directory = Path(data_directory)

        raw_mutations_files = [
            data_directory / "data_mutations_extended.txt",
            data_directory / "data_mutations.txt",
        ]
        raw_mutations = cls.try_files(
            pd.read_csv, raw_mutations_files, sep="\t", low_memory=False, comment="#"
        )

        patients = pd.read_csv(
            data_directory / "data_clinical_patient.txt", sep="\t", comment="#"
        )
        samples = pd.read_csv(
            data_directory / "data_clinical_sample.txt", sep="\t", comment="#"
        )

        expression_files = [
            data_directory / "data_RNA_Seq_v2_mRNA_median_Zscores.txt",
            data_directory / "data_mrna_seq_v2_rsem_zscores_ref_diploid_samples.txt",
            data_directory / "data_mRNA_median_Zscores.txt",
        ]
        expression = cls.try_files(pd.read_csv, expression_files, sep="\t")

        cna_files = [
            data_directory / "data_log2CNA.txt",
            data_directory / "data_log2_cna.txt",
            data_directory / "data_CNA.txt",  # This is not log2!
        ]
        cna = cls.try_files(pd.read_csv, cna_files, sep="\t", index_col=0)

        if cna is not None:
            cna = cna.drop(columns=["Entrez_Gene_Id"])

        return cls(raw_mutations, expression, cna, patients, samples)

    def select_groups(self, data_name, group1, group2):
        data = getattr(self, data_name, None)
        if data is None:
            raise ValueError

        group1_data = self.group_columns_by_any(data, group1)
        group2_data = self.group_columns_by_any(data, group2)

        data = pd.concat([group1_data, group2_data], axis=1)
        sum = data.sum(axis=1)

        neither = data.index[sum == 0]
        both = data.index[sum == 2]

        group1_only = data.index[(data.iloc[:, 0] == 0) & (data.iloc[:, 1] == 1)]
        group2_only = data.index[(data.iloc[:, 0] == 1) & (data.iloc[:, 1] == 0)]

        return neither, group1_only, group2_only, both

    def select_groups_for_sl_across(self, gene_data_name, system_genes, gene):
        system_data = self.group_columns_by_any(self.mutations, system_genes)

        gene_data = getattr(self, gene_data_name, None)
        if gene_data is None:
            raise ValueError

        gene_data = self.group_columns_by_any(gene_data, gene)

        data = pd.concat([system_data, gene_data], axis=1)
        sum = data.sum(axis=1)

        neither = data.index[sum == 0]
        both = data.index[sum == 2]

        group1_only = data.index[(data.iloc[:, 0] == 0) & (data.iloc[:, 1] == 1)]
        group2_only = data.index[(data.iloc[:, 0] == 1) & (data.iloc[:, 1] == 0)]

        return neither, group1_only, group2_only, both

    def select_groups_by_mutations(self, group1, group2):
        group1_mutations = self.group_columns_by_any(self.mutations, group1)
        group2_mutations = self.group_columns_by_any(self.mutations, group2)

        mutations = pd.concat([group1_mutations, group2_mutations], axis=1)
        sum = mutations.sum(axis=1)

        neither = mutations.index[sum == 0]
        both = mutations.index[sum == 2]

        group1_only = mutations.index[
            (mutations.iloc[:, 0] == 0) & (mutations.iloc[:, 1] == 1)
        ]
        group2_only = mutations.index[
            (mutations.iloc[:, 0] == 1) & (mutations.iloc[:, 1] == 0)
        ]

        return neither, group1_only, group2_only, both

    def get_underexpression(self, lower_pct):
        exp = self.expression.set_index("Hugo_Symbol").drop(columns="Entrez_Gene_Id")
        self.underexpression = (
            (exp < np.nanpercentile(exp, lower_pct, axis=1).reshape(-1, 1))
            .astype(int)
            .T
        )

        return self

    def filter_underexpression_with_cna(self, cna_cutoff):
        index = self.underexpression.index.intersection(self.cna.columns)
        columns = self.underexpression.columns.intersection(self.cna.index)

        cna_filtered = (self.cna < cna_cutoff).astype(int)
        underexpression = self.underexpression.loc[index, columns]
        cna_filtered = cna_filtered.T.loc[index, columns]

        self.filtered_underexpression = cna_filtered & underexpression

        return self

    def logrank_test(self, groupA_sample_ids, groupB_sample_ids):
        groupA = self.get_patient_data(sample_ids=groupA_sample_ids)
        groupB = self.get_patient_data(sample_ids=groupB_sample_ids)

        return logrank_test(
            groupA["OS_MONTHS"].values,
            groupB["OS_MONTHS"].values,
            # (groupA["OS_STATUS"].values == "0:LIVING").astype(int),
            # (groupB["OS_STATUS"].values == "0:LIVING").astype(int),
        )

    def mutual_exclusivity(self, gene1, gene2):
        aa, ab, ba, bb = self.select_groups_by_mutations(gene1, gene2)

        table = [[len(aa), len(ab)], [len(ba), len(bb)]]
        odds_ratio, pval = fisher_exact(table, alternative="less")
        return table, odds_ratio, pval

    def mutation_heatmap(self, group1, group2, remove_neither=False):
        group1_mutations = self.group_columns_by_any(self.mutations, group1)
        group2_mutations = self.group_columns_by_any(self.mutations, group2)

        mutations = pd.concat([group1_mutations, group2_mutations], axis=1)

        if remove_neither:
            mutations = mutations.loc[mutations.any(axis=1)]

        sns.clustermap(mutations, cmap="Blues")

        return mutations

    def kaplan_meier(
        self,
        patient_ids=None,
        sample_ids=None,
        metric="OS_MONTHS",
        ax=None,
        label="",
        ci_show=False,
        color="black",
        plot=True,
    ):
        data = self.get_patient_data(
            sample_ids=sample_ids,
            patient_ids=patient_ids,
        )

        data = data[metric].values
        data = data[np.isfinite(data)]

        km = KaplanMeierFitter()
        km.fit(data, label=label)

        if plot:
            if ax is None:
                ax = plt.gca()

            km.plot(
                ci_show=ci_show,
                ax=ax,
                color=color,
            )
            if metric == "OS_MONTHS":
                ax.set_xlabel("Overall survival (Months)")

            ax.set_ylabel("Surviving Fraction")

        # x = km.median_survival_time_
        # xs.append(x)
        # x0 = ax.get_xlim()[0]
        # y0 = ax.get_ylim()[0]
        # ax.plot([0, x, x], [0.5, 0.5, 0], linestyle="--", color="grey", linewidth=0.75)

        return ax, km

    def get_patient_data(self, sample_ids=None, patient_ids=None):
        if sample_ids is not None and patient_ids is not None:
            raise ValueError("Cannot give both sample_ids and patient_ids as inputs")

        if sample_ids is None and patient_ids is None:
            raise ValueError(
                "At least one of sample_ids and patient_ids must be provided"
            )

        if sample_ids is not None:
            patient_ids = [
                self.sample_patient_map[sample_id] for sample_id in sample_ids
            ]

        patient_data = self.patients.loc[self.patients["PATIENT_ID"].isin(patient_ids),]

        return patient_data

    @staticmethod
    def group_columns_by_any(dataframe, columns):
        if isinstance(columns, str):
            columns = [columns]

        _columns = dataframe.columns.intersection(columns)
        slice = dataframe[_columns]
        if len(slice.shape) != 1:
            slice = slice.any(axis=1)

        return slice

    @staticmethod
    def map_patient_samples(samples_dataframe):
        patient_sample_map = defaultdict(list)
        for _, row in samples_dataframe.iterrows():
            patient_sample_map[row["PATIENT_ID"]].append(row["SAMPLE_ID"])

        assert all([len(v) == 1 for k, v in patient_sample_map.items()])

        patient_sample_map = {k: v[0] for k, v in patient_sample_map.items()}
        sample_patient_map = {v: k for k, v in patient_sample_map.items()}

        return patient_sample_map, sample_patient_map

    @staticmethod
    def pivot_mutations_table(
        mutations, sample_col="Tumor_Sample_Barcode", gene_col="Hugo_Symbol"
    ):
        mutations_pivoted = (
            mutations[[sample_col, gene_col]]
            .drop_duplicates()
            .assign(mutations=1)
            .pivot(sample_col, gene_col, "mutations")
            .fillna(0)
            .astype(int)
        )

        return mutations_pivoted

    @staticmethod
    def remove_benign_mutations(mutations):
        """Note: I could use a combination of three metrics, but I'm just going to
        use IMPACT column for now. See this link
        https://docs.gdc.cancer.gov/Data/File_Formats/MAF_Format/#impact-categories
        for more details
        """
        return mutations.loc[mutations["IMPACT"] == "LOW"]

    @staticmethod
    def try_files(func, files, *args, **kwargs):
        for file_ in files:
            try:
                return func(file_, *args, **kwargs)

            except FileNotFoundError:
                continue


def get_decipher_sl_across():
    tissue_map = {
        "brca": "brca",
        "hnsc": "hnsc",
        "nsclc": "luad",
        "pan-cancer": None,
        "MDAMB231": "brca",
        "MCF7": "brca",
        "MCF10A": "brca",
        "CAL27": "hnsc",
        "CAL33": "hnsc",
        "A549": "luad",
        "A427": "luad",
    }

    sl_across_file = "D:/work-vault/2-Projects/decipher/decipher-dvc/data/2_scores/functions-221115/NEST/functions-by-synthetic-essential-gene-partners.csv"
    sl_across = pd.read_csv(sl_across_file)
    sl_across = sl_across.loc[
        (sl_across["fdr"] < 0.3)
        & (sl_across["ess-binom-fdr"] < 0.3)
        & (sl_across["stringency"] == "lenient")
        & (sl_across["subset"] == "complete")
        & (sl_across["context"] == "all")
        & (sl_across["condition"].isin(list(tissue_map.keys())))
    ]
    sl_across = sl_across.assign(tcga_cohort=sl_across["condition"].map(tissue_map))

    return sl_across


if __name__ == "__main__":
    decipher = Decipher()
    nest, decipher_nest, _ = filter_nest(decipher)
    nest_membership_map = {nest.namemap[k]: v for k, v in nest.membership_map.items()}

    decipher_interactions = pd.read_csv(
        "D:/work-vault/2-Projects/decipher/decipher-dvc/data/3_hits/decipher/decipher_synthetic_essential_categorized.csv"
    )
    decipher_interactions.columns = ["geneA", "geneB", "context"]

    tcga_directory = Path("D:/data/external/tcga/")
    metabric_directory = Path("D:/data/external/metabric")

    directory_map = {
        # "metabric": metabric_directory,
        "brca": tcga_directory / "brca_tcga_pan_can_atlas_2018",
        "hnsc": tcga_directory / "hnsc_tcga_pan_can_atlas_2018",
        "luad": tcga_directory / "luad_tcga_pan_can_atlas_2018",
    }

    clinical_data = []
    for name, directory in directory_map.items():
        dataset = (
            ClinicalData.from_cbioportal_directory(directory)
            .get_underexpression(50)
            .filter_underexpression_with_cna(0)
        )
        dataset.name = name

        clinical_data.append(dataset)

    brca_tcga, hnsc_tcga, luad_tcga = clinical_data

    tcga_cohorts = {
        "pan-cancer": [brca_tcga, hnsc_tcga, luad_tcga],
        "multi": [brca_tcga, hnsc_tcga, luad_tcga],
        "brca": brca_tcga,
        "hnsc": hnsc_tcga,
        "nsclc": luad_tcga,
        "A549": luad_tcga,
        "A427": luad_tcga,
        "CAL27": hnsc_tcga,
        "CAL33": hnsc_tcga,
        "MDAMB231": brca_tcga,
        "MCF7": brca_tcga,
        "MCF10A": brca_tcga,
    }
    columns = [
        "geneA",
        "geneB",
        "tcga-cohort",
        "datatype",
        "not-both-mst",
        "both-mst",
        "pval",
        "n-not-both",
        "n-both",
    ]

    sl_results = decipher_sl_in_tcga(decipher_interactions, tcga_cohorts)
    sl_results = pd.DataFrame(sl_results, columns=columns)

    decipher_sl_across = get_decipher_sl_across()
    sl_across_results = decipher_sl_across_in_tcga(
        decipher_sl_across, tcga_cohorts, nest_membership_map
    )
    sl_across_results = pd.DataFrame(sl_across_results, columns=columns)

    results = pd.concat([sl_results, sl_across_results], axis=0)
    results.to_csv("decipher-in-tcga.csv")

    """
    Missing: 
    - FDR 
    """
