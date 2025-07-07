# lexical-evolution

Code and data for *Evolution on the Lexical Workbench: Disentangling Frequency, Centrality, and Polysemy in Language Evolution*

## Description

The codebase is comprised of 3 separate steps:

1. **Sense compilation**  
   Uses a recursive algorithm to traverse the spaCy syntax tree and identify all relevant adjective–noun pairs.  
   Output is a hierarchy of directories where the first level is each unique word and the second level is each unique decade. Each subfolder contains:  
   (1) A NumPy matrix (`.npy`) containing the embeddings for all nouns the adjective modified.  
   (i.e., for the collocations `RED car`, `SMALL car`, `BIG car` in 1850, the shape of the embeddings matrix for 1850 would be `(3, d)`, where `d` is the dimensionality of the BERT embeddings).  
   (2) A two-column CSV where the first column is the relevant adjective and the second column is all of the nouns included in the embedding matrix.

2. **Semantic compilation**  
   Takes the path containing the 2-level hierarchy of word/decade directories as input, traverses the directories, and compiles semantic measures (scalar values) for all embeddings matrices.  
   Result is a wide-format CSV where each row is a word and each column is an instance of measure `x` in decade `y`.  
   (The number of columns will be equal to `n_measures * n_decades`.)

3. **Causal compilation**  
   Takes the wide-format CSV containing all words/measures and runs GMC analysis to determine causal relations.  
   In our case the null hypothesis `H₀` is that there is no unique variance in time series `y` explained by variance in time series `x`, above and beyond the unique variance in time series `x` explained by `y`.  
   Rejecting the null means that generalized measures of correlation identify a unique effect of `x` on `y`, asymmetric from the effect of `y` on `x`, and significantly different from the effect of `x` on itself.

## Installation

To get started, simply run:

```bash
./setup_env.sh
```

You will be prompted to provide a Hugging Face token in order to access BERT (needed for contextual embeddings in step 1). Then run:

```bash
conda activate lexical_evolution_env
```

Note that you need to provide the folder of COHA CSVs yourself.

## Execution

Each of the three compilation steps can be run individually in `exp_compile.py`, or sequentially in one command (`run_full_experiment`).

### Sense Compilation

```bash
python run_sense_compile.py csv_folder output_dir --decade
```

Run sense compilation for a folder of CSVs and export the two-level word–decade sense hierarchy to an output directory.  
The `--decade` flag allows optionally only running a CSV for a given decade (e.g., `--decade 1840`).

### Semantic Compilation

```bash
python run_semantic_compile.py staging_root output_csv_dir
```

Run semantic compilation for the folder of CSVs (in `staging_root` directory) and export a single wide CSV with all measures to `output_csv_dir`.