import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
# Wrod net is a english lexical library containing words similar that may be merged together into one to reduce the size of our lexicon and eventually our input feature set Because
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()
hm_lines = 100000

def create_lexicon(pos, neg):
    lexicon = []
    for fi in [pos, neg]:
        with open(fi, 'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)

    fin_lexicon = []

    for w in w_counts:
        # these values determine which words we skip out of the complete lexicon
        if 1000 > w_counts[w] > 50:
            fin_lexicon.append(w)

    print(len(fin_lexicon))
    # so now our final lexicon is an array filled with our lower case unique words which lie within our threshold
    return fin_lexicon

def create_training_examples(lexicon, sample, classification):
    train_example = []
    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            tokens = word_tokenize(l.lower())
            tokens = [lemmatizer.lemmatize(i) for t in tokens]
            # here we check if any of our tokens are in the fin_lexicon
            example = np.zeros(len(lexicon))
            for t in tokens:
                if t.lower() in lexicon:
                    example[lexicon.index(t.lower)] += 1


            train_example.append([list(example), classification])

    return train_example

def create_feature_sets_and_labels(pos, neg, t_size = 0.6):
    features = []

    lexicon = create_lexicon(pos, neg)

    pos_feat = create_training_examples(lexicon, pos, [1, 0])
    neg_feat = create_training_examples(lexicon, neg, [0, 1])

    features.append(pos_feat)
    features.append(neg_feat)

    random.shuffle(features)

    features = np.array(features)

    # now i want to divide my features in train, cv and test_size
    train_size = int(t_size * len(features))

    train_features =    list(features[:train_size, 0])
    train_labels   =    list(features[:train_size, 1])

    text_features  =    list(features[train_size+1:, 0])
    text_labels    =    list(features[train_size+1:, 1])

    return train_features, train_labels, text_features, text_labels



if __name__ == "__main__":
    train_features, train_labels, text_features, text_labels = create_feature_sets_and_labels('pos.txt', 'neg.txt', 0.6)
    with open('sentiment-set.pickle', 'wb') as f:
        pickle.dump([train_features, train_labels, text_features, text_labels], f)
