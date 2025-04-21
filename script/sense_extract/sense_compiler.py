#!/usr/bin/env python

import pandas as pd
import spacy
import en_core_web_sm
from tqdm import tqdm
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import numpy as np
nlp = spacy.load('en_core_web_sm')#load tokenizer
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from concurrent.futures import ProcessPoolExecutor
from script.utils import debug_utils

class RawSenseCompiler:
    

    def __init__(self, csv_paths):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = TFBertModel.from_pretrained('bert-base-uncased')
        self.source_dfs = []

        self.ref_df = pd.read_csv('./data/antonyms/antonyms_words.csv')

        for path in csv_paths: #read in all CSV to source_df
            df = pd.read_csv(path).head(1000)
            df.name = path
            self.source_dfs.append(df)

        for df in self.source_dfs:
            raw_word_vecs = self.compile_individual_raw_phrase_vecs(df)
            
            print(f"Total antonym pair CSV coverage for {df.name}")
            print(f"Total ref CSV coverage: {len(debug_utils.csv_coverage(self.ref_df, raw_word_vecs))}/{len(self.ref_df) * 2}")

            self.embeddings = self.compile_word_phrase_embeddings(raw_word_vecs, df.name[-8:-4])#get the last 4 chars for date


    
    def __get_noun_adj_pair(self, search_word, raw_sentence):
        """
        Get pairs of noun targeted by adjective in a sentence
        Args:
            search_word (str): adjective to find targeted noun for
            raw_sentence (str): single sentence string
        Returns:
            dict: {adj: index, noun: index}: key value pair of adj, noun, and corresponding tokenized indices for each
        """
        tokenized = nlp(raw_sentence)

        adj_token = None
        for token in tokenized:#get adjective/search_word index
            if token.text.lower() == search_word.lower() and token.pos_ == 'ADJ':
                adj_token = token
                break
        
        if adj_token:#now get noun index
            noun_token = self.__tree_recursive_spacy(adj_token)
            if noun_token:
                return {
                    'sentence': raw_sentence,
                    'tok_sentence': tokenized,
                    'adj': {'word': adj_token, 'index': adj_token.i},  # Index of the adjective in the tokenized sentence
                    'noun': {'word': noun_token, 'index':noun_token.i} # Index of the noun in the tokenized sentence
                }
        
        # If none found return empty dict
        return None
    

    
    def __tree_recursive_spacy(self, token, depth =0, max_depth=50):
        """
        Recursively search the dependency DAG structure for noun-adjective dependencies
        """
        if(token == None):
            return None

        if(depth>max_depth):
            return None#error handling for stack overflow

        if token.dep_ == 'amod' or (token.dep_ == 'conj' and token.head.pos_ == 'NOUN'):
            return token.head# case 1: if parent connection is adjective modifier, return parent
        
        for child in token.children:
            if child.dep_ in ['conj', 'amod', 'nsubj']:
                if child.pos_ == 'NOUN' or child.pos_ == 'PRON':
                    return child#case 2: if direct child connection to noun exists
            if child.dep_ in ['advcl']:
                return self.__tree_recursive_spacy(token.head, depth=depth+1)        

        
        if token.dep_ in ['conj', 'acomp', 'nsubj', 'oprd', 'advcl', 'acl']: #recursive case: if parent connection of these kinds exists
            return self.__tree_recursive_spacy(token.head, depth=depth+1)

        return None#no dependency found


    
    def compile_word_phrase_embeddings(self, word_vec, decade_name):
        """
        For a key/value pair of word and associated phrase vector, convert to a matrix of vector embeddings.
        Each column represents an embedding of a phrase, and the number of rows corresponds to embedding dimensions (768).
        Args:
            word_vec (dict): {word: [phrases[str]]}
            decade_name (str): Name of the decade, used to create subfolders within each word's folder
        """
        word_embeddings = {}

        # Create a root directory for exports if it doesn't already exist
        root_dir = 'exports'
        os.makedirs(root_dir, exist_ok=True)

        for word, phrases in tqdm(word_vec.items(), desc="Compiling embeddings for all words"):
            unique_phrases = set(phrases)
            embeddings_list = []

            for phrase in unique_phrases:
                inputs = self.tokenizer(phrase, return_tensors="tf", add_special_tokens=True)
                outputs = self.model(**inputs)
                
                phrase_embedding = outputs.last_hidden_state[0, 1].numpy()
                embeddings_list.append(phrase_embedding)

            if embeddings_list:
                embeddings_array = np.column_stack(embeddings_list)
                word_embeddings[word] = embeddings_array

                # Adjust directory path to include the adjective first, then decade
                word_dir = os.path.join(root_dir, word, decade_name)
                os.makedirs(word_dir, exist_ok=True)

                # Save numpy array
                npy_filename = os.path.join(word_dir, f'{word}_embeddings.npy')
                np.save(npy_filename, embeddings_array)

                # Save CSV for metadata
                csv_filename = os.path.join(word_dir, f'{word}_metadata.csv')
                pd.DataFrame({'Noun Phrases': list(unique_phrases)}).to_csv(csv_filename, index=False)

                print(f"Saved embeddings to {npy_filename}")
                print(f"Saved metadata to {csv_filename}")
            else:
                word_embeddings[word] = np.array([], shape=(768, 0))

        return word_embeddings
            
    def compile_individual_raw_phrase_vecs(self, df):
        """
        Compile raw phrase vectors from the provided dataframe.
        """
        adj_noun_pairs_dict = {}
        for sentence in tqdm(df.itertuples(), desc=f"Processing {df.name}", total=len(df)):
            pairs_dict = self.__get_noun_adj_pair(sentence.search_word, sentence.sentence)
            if pairs_dict is not None:
                adj = str(pairs_dict['adj']['word'])
                noun = str(pairs_dict['noun']['word'])
                pair_phrase = adj + " " + noun
                if adj not in adj_noun_pairs_dict:
                    adj_noun_pairs_dict[adj] = []
                adj_noun_pairs_dict[adj].append(pair_phrase)
        return adj_noun_pairs_dict
    
def process_file(csv_path):
        print("PROC FILE:")
        print(csv_path)
        compiler = RawSenseCompiler([csv_path])
        return compiler

if __name__ == '__main__':
    src_csvs = ['./data/coha/fic/coha_words_fic_1820.csv']
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_file, src_csvs))
    compiler = RawSenseCompiler(src_csvs)