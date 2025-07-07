#!/usr/bin/env python

import pandas as pd
import spacy
from tqdm import tqdm
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import csv
import numpy as np
import os
import sys

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Set TensorFlow to grow GPU memory as needed.")
    except RuntimeError as e:
        print(e)

nlp = spacy.load('en_core_web_sm')  # Load tokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import debug_utils


class RawSenseCompiler:
    def __init__(self, csv_paths, output_dir="exports"):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = TFBertModel.from_pretrained('bert-base-uncased')
        self.source_dfs = []
        self.output_dir = output_dir

        self.ref_df = pd.read_csv('./data/metadata/stimuli.csv')

        for path in csv_paths:
            print("PATH")
            print(path)
            df = pd.read_csv(
                path,
                on_bad_lines='skip',
                engine='python',
                encoding='utf-8'
            )
            df.name = path
            self.source_dfs.append(df)

        for df in self.source_dfs:
            raw_word_vecs = self.compile_individual_raw_phrase_vecs(df)
            print(f"Total antonym pair CSV coverage for {df.name}")
            print(f"Total ref CSV coverage: {len(debug_utils.csv_coverage(self.ref_df, raw_word_vecs))}/{len(self.ref_df) * 2}")
            self.embeddings = self.compile_word_phrase_embeddings(raw_word_vecs, df.name[-8:-4])

    def __get_noun_adj_pair(self, search_word, raw_sentence):
        tokenized = nlp(raw_sentence)
        adj_token = next((t for t in tokenized if t.text.lower() == search_word.lower() and t.pos_ == 'ADJ'), None)
        if adj_token:
            noun_token = self.__tree_recursive_spacy(adj_token)
            if noun_token:
                return {
                    'sentence': raw_sentence,
                    'tok_sentence': tokenized,
                    'adj': {'word': adj_token, 'index': adj_token.i},
                    'noun': {'word': noun_token, 'index': noun_token.i}
                }
        return None

    def __tree_recursive_spacy(self, token, depth=0, max_depth=50):
        if token is None or depth > max_depth:
            return None
        if token.dep_ == 'amod' or (token.dep_ == 'conj' and token.head.pos_ == 'NOUN'):
            return token.head
        for child in token.children:
            if child.dep_ in ['conj', 'amod', 'nsubj'] and child.pos_ in ['NOUN', 'PRON']:
                return child
            if child.dep_ == 'advcl':
                return self.__tree_recursive_spacy(token.head, depth + 1)
        if token.dep_ in ['conj', 'acomp', 'nsubj', 'oprd', 'advcl', 'acl']:
            return self.__tree_recursive_spacy(token.head, depth + 1)
        return None

    def compile_word_phrase_embeddings(self, word_vec, decade_name):
        word_embeddings = {}
        root_dir = self.output_dir
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

                word_dir = os.path.join(root_dir, word, decade_name)
                os.makedirs(word_dir, exist_ok=True)

                np.save(os.path.join(word_dir, f'{word}_embeddings.npy'), embeddings_array)
                pd.DataFrame({'Noun Phrases': list(unique_phrases)}).to_csv(
                    os.path.join(word_dir, f'{word}_metadata.csv'), index=False)

                print(f"Saved embeddings to {word_dir}/{word}_embeddings.npy")
                print(f"Saved metadata to {word_dir}/{word}_metadata.csv")
            else:
                word_embeddings[word] = np.array([], shape=(768, 0))

        return word_embeddings

    def compile_individual_raw_phrase_vecs(self, df):
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


def run_sense_compiler(csv_paths, output_dir="exports"):
    """
    Runs the RawSenseCompiler on a list of CSV paths with optional output dir.
    """

    compiler = RawSenseCompiler(csv_paths, output_dir=output_dir)
    return compiler
