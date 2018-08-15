import pandas as pd
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import re

from matplotlib.axes import Axes
import matplotlib.font_manager as fm
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from functools import partial
import re


def read_glove(a_path):
    return pd.read_table(a_path, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE,
                      na_values=None, keep_default_na=False)


def vec(w, words):
    return words.loc[w.lower()].values


def ind(w, words):
    return words.loc[w].index


def slip(sent, words, dist=1):
    new_sent = []
    for a_word in sent:
        nws = filter_words(find_closest_word(a_word, words).values)
        if len(nws) > dist:
            new_sent.append(nws[dist])
        else:
            new_sent.append(a_word)
    return new_sent


def find_closest_to_vec(v, words, start_ind=1, end_ind=8):
    diff = words - v
    delta = np.sum(diff * diff, axis=1)
    i = delta.argsort()[start_ind:end_ind]
    return words.iloc[i].index


def find_closest_word(w, words, start_ind=1, end_ind=15):
    return find_closest_to_vec(v=vec(w, words), words=words, start_ind=start_ind, end_ind=end_ind)


def constellate3d(ws, words, fignum=1):
    fig = plt.figure(fignum, figsize=(8, 11), dpi=150,)
    plt.clf()
    ax = Axes3D(fig)
    vs = list(map(partial(vec, words=words), ws))
    pca = decomposition.PCA(n_components=3)
    pca.fit(vs)
    X = pca.transform(vs)
    for i in range(len(X)):  # plot each point + it's index as text above
        ax.scatter(X[i, 0], X[i, 1], X[i, 2], s=1)
        ax.text(X[i, 0], X[i, 1], X[i, 2], '%s' % (ws[i]), size=8, zorder=1,
                color='k')
    plt.show()


def constellate(ws, words, fignum=1, family=None):
    fig = plt.figure(fignum, figsize=(8,6), dpi=150,)
    plt.clf()
    vs = list(map(partial(vec, words=words), ws))
    pca = decomposition.PCA(n_components=2)
    pca.fit(vs)
    X = pca.transform(vs)

    plt.plot(X[:, 0], X[:, 1], linewidth=0.25)

    for i in range(len(X)):  # plot each point + it's index as text above
        plt.scatter(X[i, 0], X[i, 1], s=2)
        if family:
            plt.text(X[i, 0] - len(ws[i])*0.02, X[i, 1] - 0.15, '%s' % (ws[i]), size=11, zorder=1,
                color='k', fontproperties=family)
        else:
            plt.text(X[i, 0], X[i, 1], '%s' % (ws[i]), size=11, zorder=1,
                color='k')
    plt.show()


def get_words():
    glove_data_folder = os.path.join("/", "home", "qfwfq", "Tinkering", "GloVe")
    wiki_data_file = os.path.join(glove_data_folder, "Wikipedia", "glove.6B.200d.txt")
    crawl_data_file = os.path.join(glove_data_folder, "CommonCrawl", "glove.42B.300d.txt")
    return read_glove(wiki_data_file)


def filter_words(some_words):
    return list(filter(lambda w: re.match(r'^[a-zA-Z]+$', w), some_words))


def main():
    words = get_words()

    mf1 = fm.FontProperties(fname='fonts/mfp1.otf')
    mf2 = fm.FontProperties(fname='fonts/mfp2.otf')
    mf3 = fm.FontProperties(fname='fonts/mfp3.otf')
    mf4 = fm.FontProperties(fname='fonts/mfp4.otf')
    mf5 = fm.FontProperties(fname='fonts/mfp5.otf')
    mf6 = fm.FontProperties(fname='fonts/mfp6.otf')
    mf7 = fm.FontProperties(fname='fonts/mfp7.otf')

    fonts = [mf1,mf2,mf3,mf4,mf5,mf6,mf7,mf1,mf1,mf1,mf1,mf1,mf1,mf1]

    # print(find_closest_word("hello", words))
    # print(find_closest_word("darkness", words))
    # print(find_closest_to_vec(vec("hello", words) - vec("world", words) + vec("darkness", words), words))
    # print(find_closest_to_vec(vec("alive", words) - vec("dead", words) + vec("person", words), words))
    # print(find_closest_to_vec(vec("black", words) - vec("white", words) + vec("brown", words), words))


    a_palmfull_of_words = ["darkness", "hello", "goodbye", "shadow", "thread", "terrain", "railroad", "orientation", "weave", "cut", "slice", "line", "enter", "embed", "path"]
    some_sentence_says = ["writing", "in", "the", "chambers", "a", "different", "piss", "unfamiliar","single", "stringed", "instrument"]
    near = ["violet", "indigo", "blue", "green", "yellow", "orange", "red"]
    apparition = ["the","apparition","of","these","faces","in","the","crowd",";","petals","on","a","wet","black","bough"]
    sidewalk = ["i","was","following","the","center","of","the","sidewalk","'s","triangle","over","the","orange","and","concrete"]
    last = ["last", "words", "suspended", "into", "among", "subtract", "backpropagation", "cell", "hairy", "stuttered", "spat"]
    mazes = "to start in mazes becoming surfaces just labyrinths exhausting quiet voyagers walking past destinations".split(" ")
    marching = "taking beginnings actively indicting fields marching jocularly piloted quips wax blaze".split(" ")

    slippages = map(lambda i: (slip(marching, words, i)), range(7))

    # constellate(a_palmfull_of_words, words, 1)
    # constellate(some_sentence_says, words, 2)
    # constellate(near, words, 3)
    # constellate(apparition, words, 4)
    # constellate(sidewalk, words, 5)
    # constellate(last, words, 7)

    i = 2
    constellate(marching, words, 1, family=mf1)

    for sent in list(slippages):
        print(sent)
        constellate(sent, words, i, fonts[i-1])
        i += 1


if __name__ == "__main__":
    main()