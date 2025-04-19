# lexical-evolution

Code and data for *Evolution on the Lexical Workbench: Disentangling Frequency, Centrality, and Polysemy in Language Evolution*

## Scripts

- exp_compile.py
    - compile individual experiment/measure components, or run master experiment

### sense_extract
- sense_extract.py
    - v1
        -  extract adjective word senses from COHA (fic), 1820 - 2000

### measures
- network/network_compile.py
    - v1
        - compile network properties of all words using sliding window
- frequency_compile.py
    - v1
        - get word frequency measure
- semantic/semantic_compile.py
    - v1
        - compile all semantic properties of words
            - spectral diversity
            - entropy
            - nonzero eigenvalues
            condition number
### causality
- gmc.ipynb
    - v1
        - Run pairwise R code for GMC(generalized measures of correlation) analysis on network, frequency, semantic properties across all words in all decades
    
- general_measures.ipynb
    - v1
        - Get general statistical properties of measures and relationships, as well as correlations




