import pathlib

import numpy as np
import scipy.sparse
import scipy.io
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
import utils.preprocess
from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as sklearn_stopwords
from nltk import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from utils.data import load_IMDB_data
from utils.data import load_glove_vectors

nx_G_lists, edge_metapath_indices_lists, features_list, adjM, type_mask, labels, train_val_test_idx = load_IMDB_data()
expected_metapaths = [
    [(0, 1, 0), (0, 1, 2, 1, 0), (0, 1, 3, 1, 0)],
    [(1, 0, 1), (1, 2, 1), (1, 3, 1)],
    [(2, 1, 2), (2, 1, 0, 1, 2), (2, 1, 3, 1, 2)],
    [(3, 1, 3), (3, 1, 0, 1, 3), (3, 1, 2, 1, 3)]
]

for i in range(1):
    # get metapath based neighbor pairs
    neighbor_pairs = utils.preprocess.get_metapath_neighbor_pairs(adjM, type_mask, expected_metapaths[i])
    # construct and save metapath-based networks
    G_list = utils.preprocess.get_networkx_graph(neighbor_pairs, type_mask, i)

    # save data
    # networkx graph (metapath specific)
    for G, metapath in zip(G_list, expected_metapaths[i]):
        nx.write_adjlist(G, save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '.adjlist')
    # node indices of edge metapaths
    all_edge_metapath_idx_array = utils.preprocess.get_edge_metapath_idx_array(neighbor_pairs)
    for metapath, edge_metapath_idx_array in zip(expected_metapaths[i], all_edge_metapath_idx_array):
        np.save(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.npy', edge_metapath_idx_array)
# save data
# all nodes adjacency matrix
scipy.sparse.save_npz(save_prefix + 'adjM.npz', scipy.sparse.csr_matrix(adjM))
# all nodes (authors, papers, terms and conferences) features
# currently only have features of authors, papers and terms
scipy.sparse.save_npz(save_prefix + 'features_{}.npz'.format(0), features_author)
scipy.sparse.save_npz(save_prefix + 'features_{}.npz'.format(1), features_paper)
np.save(save_prefix + 'features_{}.npy'.format(2), features_term)
# all nodes (authors, papers, terms and conferences) type labels
np.save(save_prefix + 'node_types.npy', type_mask)
# author labels
np.save(save_prefix + 'labels.npy', labels)
# author train/validation/test splits
rand_seed = 1566911444
train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=400, random_state=rand_seed)
train_idx, test_idx = train_test_split(train_idx, test_size=3257, random_state=rand_seed)
train_idx.sort()
val_idx.sort()
test_idx.sort()
np.savez(save_prefix + 'train_val_test_idx.npz',
         val_idx=val_idx,
         train_idx=train_idx,
         test_idx=test_idx)