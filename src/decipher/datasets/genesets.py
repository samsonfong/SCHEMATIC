"""Geneset data"""

from pathlib import Path
from decipher import PROJECT_BASE


def read_geneset_by_name(name):
    genesets_path = Path(PROJECT_BASE / "data/msigdb-genesets")

    geneset_filemap = {
        "kegg": "c2.cp.kegg.v7.4.symbols.gmt",
        "reactome": "c2.cp.reactome.v7.2.symbols.gmt",
        "wikipathways": "c2.cp.wikipathways.v7.4.symbols.gmt",
        "go:bp": "c5.go.bp.v7.4.symbols.gmt",
    }

    genesets = read_gmt_file(genesets_path / geneset_filemap[name])

    if name == "kegg":
        genesets["KEGG_HOMOLOGOUS_RECOMBINATION"].append("BRCA1")  # Don't know why...

    return genesets


def read_gmt_file(filename):
    """Read msigdb file format"""

    geneset = {}
    with open(filename) as f:
        for line in f.readlines():
            array = line.split("\t")
            array = [i.strip() for i in array]
            geneset[array[0]] = array[2:]

    return geneset
