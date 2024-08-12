from decipher import PROJECT_BASE
from pathlib import Path
import pandas as pd
from collections import namedtuple


class Cosmic(object):
    def __init__(
        self,
        cosmic_file=None,
    ):
        if cosmic_file is None:
            cosmic_file = PROJECT_BASE / "data/cosmic_cancer_gene_census_201214.csv"
        else:
            cosmic_file = Path(cosmic_file)

        self.data = pd.read_csv(cosmic_file)
        self.tsg = self.data[self.data["Role in Cancer"] == "TSG"]["Gene Symbol"].values

    def is_tsg(self, gene):
        if isinstance(gene, str):
            return gene in self.tsg

        # Don't bother testing if it's not a string (for instance, list of string will just return as no)
        return False

    def designate_alterations_and_ko(self, geneA, geneB):
        Genepair = namedtuple("genepair", ["alteration", "crispr_ko"])

        geneA_is_tsg = self.is_tsg(geneA)
        geneB_is_tsg = self.is_tsg(geneB)

        if geneA_is_tsg and not geneB_is_tsg:
            return [Genepair(alteration=geneA, crispr_ko=geneB)]

        elif not geneA_is_tsg and geneB_is_tsg:
            return [Genepair(alteration=geneB, crispr_ko=geneA)]

        elif geneA_is_tsg and geneB_is_tsg:
            return [
                Genepair(alteration=geneA, crispr_ko=geneB),
                Genepair(alteration=geneB, crispr_ko=geneA),
            ]

        else:
            return []
