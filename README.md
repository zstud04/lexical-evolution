# lexical-evolution

Code and data for *Evolution on the Lexical Workbench: Disentangling Frequency, Centrality, and Polysemy in Language Evolution*

## Description

The codebase is comprised of 3 separate steps:
    1. **Sense compilation**
        Uses a recursive algorithm to traverse the spacey syntax tree and identify all relevant adjective-noun pairs. Output is a hierarchy of directories where the first level is each unique word and second level is each unique decade. Each subfolder contains:
            (1). A numpy matrix (.npy) containing the embeddings for all nouns the adjective modified. (i.e for the collocations RED car, SMALL car, BIG car in 1850, the shape of the embeddings matrix for 1850 would be (3, d) where d is the dimensionality of the BERT embeddings). 
            (2). A two column csv where the first column is the relevant adjective and the second column is all of the nouns included in the embedding matrix.
    2. **Semantic compilation**
        Takes the path containing the 2 level hierarchy of word/decade directories as input, traverse the directories and compile semantic measures (scalar values) for all embeddings matrices. Result is a wide format CSV where each row is a word and each column is an instance of measure x in decade y. (number of cols will be equal to n. measures * n. decades)
    3. **Causal compilation**
        Takes the wide format CSV containing all words/measures and runs GMC analysis to determine causal relations. In our case the null hypothesis H_0 is that there is no unique variance in time series y explained by variance in time series x, above and beyond unique variance in time series x explained by time series y. Rejecting the null means that generalized measures of correlation identify a unique effect of x on y, asymmetric from the effect of y on x, and significantly different from the effect of x on itself.



## Installation

To get started, simply run ./setup_env.sh. You will be prompted to provide a huggingface token in order to access BERT (needed for contextual embeddings in step 1). Then run conda activate lexical_evolution_env. Note that you need to provide the folder of COHA CSVs yourself.

## Execution
Each of the three compilation steps can be run individually in exp_compile.py, or sequentially in one command (run_full_experiment). 

### python run_sense_compile.py csv_folder output_dir --decade
Run sense compilation for a folder of csvs and export the two level word-decade sense hierarchy to an output directory. --decade allows for optionally only running a csv for a given decade(eg. --decade 1840)

### python run_semantic_compile.py staging_root output_csv_dir
Run semantic compilation for the folder of csvs(in stage root directory) and export a single wide csv with all measures to output_csv_dir





