from multiprocessing import Pool
from decipher.scores import bagel
from decipher.datasets.decipher_data import Decipher
from decipher import PROJECT_BASE
import pandas as pd
from pathlib import Path

BASE = PROJECT_BASE / f"data/decipher/bagel"
decipher = Decipher()

bagel_path = PROJECT_BASE / "bin/bagel"
essential_genes = pd.read_csv(bagel_path / "CEGv2.txt", sep="\t")["GENE"].values

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

    bagel.prepare_nonessentials(input_files, axis=axis)
    bagel.prepare_essentials(input_files, decipher, essential_genes, axis=axis)

    combo_map = decipher.combo_map.copy()
    combo_map.update({cell_line: [cell_line] for cell_line in decipher.cell_lines})

    kwargs = [(i, j, output_files, axis) for i, j in combo_map.items()]

    with Pool(16) as pool:
        pool.starmap(bagel.run_bagel, kwargs)
