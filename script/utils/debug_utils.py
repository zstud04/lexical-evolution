"""
Various methods for debugging the coverage rate for adjectives identified in corpora
"""

import pandas as pd

def csv_coverage(src_df, ref_dict:dict):
    """
    Check the synonym antonym coverage for a dict of word senses for a given decade
    """

    included_words = []

    for word in pd.concat([src_df['a'],src_df['b']]):
        #print(word)
        if word in ref_dict.keys():
            included_words.append(word)
    
    return included_words