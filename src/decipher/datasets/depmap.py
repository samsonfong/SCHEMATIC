"""DepMap related code"""

from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


class DepMap(object):
    def __init__(self, datapath=None):
        if datapath is None:
            self.set_datapath_from_config()

        else:
            self.datapath = Path(datapath)

        if not self.datapath.exists():
            raise ValueError(f"Given datapath ({self.datapath}) does not exist!")

        self._index_type = "id"

    def set_datapath_from_config(self):
        import yaml

        from .. import PROJECT_BASE

        with open(PROJECT_BASE / "conf/datasets.yaml") as f:
            datapath = yaml.safe_load(f)["filepaths"]["depmap"]

        self.datapath = PROJECT_BASE / datapath

    def load(
        self,
        mutations=False,
        gene_effect=False,
        expression=False,
        copy_number=False,
        # common_essentials=False,
        robust_copy_number=False,
    ):
        """Load select data types"""

        # sample_info is mandatory to load
        self.load_sample_info()

        if mutations:
            self.load_mutations()

        if gene_effect:
            self.load_gene_effect()

        if expression:
            self.load_expression()

        # if common_essentials:
        #     self.load_common_essentials()

        if copy_number:
            self.load_copy_number()

        if robust_copy_number:
            self.load_robust_copy_number()

        self.use_cell_line_names_as_id()

        return self

    def load_all(self):
        """Load all data types"""

        return self.load(
            mutations=True,
            gene_effect=True,
            expression=True,
            copy_number=True,
            # common_essentials=True,
            robust_copy_number=True,
        )

    def load_sample_info(self, filename="Model.csv"):
        """Load cell line information"""

        self.sample_info = pd.read_csv(self.datapath / filename).set_index("ModelID")
        self.id_name_map = self.sample_info["StrippedCellLineName"].to_dict()
        self.name_id_map = {v: k for k, v in self.id_name_map.items()}

        self.lineage = self.sample_info["SampleCollectionSite"].unique()

        return self.sample_info

    def load_mutations(self, filename="OmicsSomaticMutations.csv"):
        """Load cell line mutations"""

        self.mutations = pd.read_csv(self.datapath / filename, low_memory=False)

        self.sq_mutations = (
            self.mutations.loc[
                self.mutations["VariantInfo"] != "SILENT", ["HugoSymbol", "ModelID"]
            ]
            .drop_duplicates()
            .assign(_mutation=1)
            .pivot(index="ModelID", columns="HugoSymbol", values="_mutation")
            .fillna(0)
            .astype(bool)
        )

        return self.mutations

    def load_gene_effect(self, filename="CRISPRGeneEffect.csv"):
        """Load single knockout data"""

        self.gene_effect = pd.read_csv(self.datapath / filename, index_col=0)
        self.gene_effect.columns = self._fix_depmap_gene_names(self.gene_effect.columns)

        return self.gene_effect

    def load_expression(self, filename="OmicsExpressionProteinCodingGenesTPMLogp1.csv"):
        """Load expression"""

        self.expression = pd.read_csv(
            self.datapath / filename, index_col=0
        )  # First column is missing "DepMap_ID" for some reason
        self.expression.index.name = "DepMap_ID"

        self.expression.columns = self._fix_depmap_gene_names(self.expression.columns)

        return self.expression.columns

    def load_copy_number(self, filename="OmicsCNGene.csv"):
        """Load copy number"""

        self.copy_number = pd.read_csv(
            self.datapath / filename, index_col=0
        )  # First column is missing "DepMap_ID" for some reason
        self.copy_number.index.name = "DepMap_ID"

        self.copy_number.columns = self._fix_depmap_gene_names(self.copy_number.columns)

        return self.copy_number.columns

    def load_robust_copy_number(self, underexpression_pct=25):
        under_expression = self.expression < np.percentile(
            self.expression, underexpression_pct, axis=0
        )
        under_copy_number = (self.copy_number < 1).astype(bool)

        self.robust_copy_number = under_expression & under_copy_number

        return self.robust_copy_number

    def load_common_essentials(self, filename="CRISPRInferredCommonEssentials.csv"):
        """Load list of common essential genes"""

        self.common_essentials = pd.read_csv(self.datapath / filename).values.ravel()
        self.common_essentials = self._fix_depmap_gene_names(self.common_essentials)

        return self.common_essentials

    def load_logfold_change(
        self, filename="AvanaLogfoldChange.csv", guidemap_filename="AvanaGuideMap.csv"
    ):
        """Load logfold change"""

        self.logfold_change = pd.read_csv(self.datapath / filename, index_col=0)

        if guidemap_filename is None:
            return self.logfold_change

        guidemap = pd.read_csv(self.datapath / guidemap_filename, index_col=0)
        guidemap = guidemap.loc[
            guidemap["Gene"].str.endswith(")").fillna(False)
            & (guidemap["UsedByChronos"]),
            "Gene",
        ].to_dict()
        guidemap = {k: v.split("(")[0].strip() for k, v in guidemap.items()}
        self.logfold_change["Gene"] = self.logfold_change.index.map(guidemap)

        return self.logfold_change

    def use_cell_line_names_as_id(self):
        """Set expression and gene_effect DataFrame index to cell line names"""

        attrs = ["expression", "gene_effect", "copy_number", "robust_copy_number"]

        for attr in attrs:
            df = getattr(self, attr, None)
            if df is None:
                continue

            df.index = self.convert_id_to_name(df.index)

        self._index_type = "name"

    def filter_samples_by_lineage(self, lineage, samples=None, depmap_id=False):
        """Get samples that belong to a lineage"""

        column = "DepMap_ID" if depmap_id else "StrippedCellLineName"
        valid_samples = self.sample_info.loc[
            self.sample_info["OncotreeLineage"] == lineage, column
        ].values

        if samples is not None:
            return np.setdiff1d(valid_samples, samples)

        return valid_samples

    def calculate_sensitivity(
        self,
        ko_gene,
        alterations,
        alteration_data_name="sq_mutations",
        min_samples=3,
        lineage=None,
    ):
        mutated_lines, wt_lines = self.separate_cell_lines(
            alterations, alteration_data_name
        )
        if lineage is not None:
            samples = self.filter_samples_by_lineage(lineage)
            mutated_lines = np.intersect1d(mutated_lines, samples)
            wt_lines = np.intersect1d(wt_lines, samples)

        mutated_lines = np.intersect1d(mutated_lines, self.gene_effect.index)
        wt_lines = np.intersect1d(wt_lines, self.gene_effect.index)

        mutated_effects = self.gene_effect.loc[mutated_lines, ko_gene].dropna().values
        wt_effects = self.gene_effect.loc[wt_lines, ko_gene].dropna().values

        if (len(mutated_effects) < min_samples) or (len(wt_effects) < min_samples):
            return

        diff = np.median(mutated_effects) - np.median(wt_effects)
        stat, pval = ttest_ind(mutated_effects, wt_effects, alternative="less")

        results = Sensitivity_Results(diff, stat, pval, mutated_effects, wt_effects)

        return results

    def separate_cell_lines(self, alterations, alteration_data_name="sq_mutations"):
        alteration_data = getattr(self, alteration_data_name)
        alteration_data = alteration_data[alterations]

        if isinstance(alteration_data, pd.DataFrame):
            alteration_data = alteration_data.any(axis=1)

        mutated_lines, wt_lines = separate_index(alteration_data)

        return mutated_lines, wt_lines

    def split_samples_by_mutations(
        self,
        mutated_genes,
        gene_column="Hugo_Symbol",
        deleterious_only=True,
        hotspot_only=False,
    ):
        """Separate profiled cell lines based on mutation status"""

        data = self.mutations
        if deleterious_only:
            data = data.loc[data["isDeleterious"] == True]

        if hotspot_only:
            data = data.loc[data["isCOSMIChotspot"] == True]

        try:
            mutated = data[gene_column].isin(mutated_genes)
        except TypeError:  # mutated_genes might not be a list
            mutated = data[gene_column].isin([mutated_genes])

        mutated = data.loc[mutated, "DepMap_ID"].unique()
        all_lines = self.mutations["DepMap_ID"].unique()
        not_mutated = np.setdiff1d(all_lines, mutated)

        if self._index_type == "name":
            mutated = self.convert_id_to_name(mutated)
            not_mutated = self.convert_id_to_name(not_mutated)

        return mutated, not_mutated

    def split_samples_by_expression(self, gene, lower_pct=50, upper_pct=50):
        """Separate profiled cell lines based on expression"""

        data = self.expression[gene].values
        under_index = np.argwhere(data < np.percentile(data, lower_pct)).ravel()
        over_index = np.argwhere(data > np.percentile(data, upper_pct)).ravel()

        under = np.array(self.expression.index[under_index])
        over = np.array(self.expression.index[over_index])

        if self._index_type == "name":
            under = self.convert_id_to_name(under)
            over = self.convert_id_to_name(over)

        return under, over

    def get_unique_vulnerabilities(self, sample, cutoff=-1):
        """Get unique vulnerabilities"""

        vulnerabilities = self.gene_effect.columns[
            self.gene_effect.loc[sample] < cutoff
        ]
        return np.setdiff1d(vulnerabilities, self.common_essentials)

    def query_gene_effect(self, gene_ko, group):
        """Get single-gene knockout of a group"""

        data = self._safe_index(self.gene_effect, group)

        return data[gene_ko]

    def convert_id_to_name(self, ids):
        return np.array([self.id_name_map.get(i, i) for i in ids])

    def convert_name_to_id(self, names):
        return np.array([self.name_id_map.get(i, i) for i in names])

    def safe_index(self, attr, index=None, columns=None):
        data = getattr(self, attr)

        if index is not None:
            data = self._safe_index(data, index)

        if columns is not None:
            data = self._safe_columns(data, columns)

        return data

    @staticmethod
    def _fix_depmap_gene_names(columns):
        return np.array([column.split()[0].strip() for column in columns])

    @staticmethod
    def _safe_index(dataframe, index):
        """Performs intersection prior to indexing"""

        safe_index = dataframe.index.intersection(index)
        # if len(safe_index) == 0:
        #     raise ValueError("No overlap found between index and dataframe")

        return dataframe.loc[safe_index]

    @staticmethod
    def _safe_columns(dataframe, columns):
        """Performs intersection prior to indexing columns"""

        safe_index = dataframe.columns.intersection(columns)
        # if len(safe_index) == 0:
        #     raise ValueError("No overlap found between index and dataframe")

        return dataframe[safe_index]


def separate_index(series):
    true_index = series.index[series.values]
    false_index = series.index[~series.values]

    return true_index, false_index


Sensitivity_Results = namedtuple(
    "Sensitivity_Results",
    ["difference", "tstat", "pvalue", "mutated_effects", "wildtype_effects"],
)
