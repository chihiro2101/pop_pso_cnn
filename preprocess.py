import math
from nltk.stem import PorterStemmer
import numpy as np
from nltk.corpus import stopwords
import os
import re
import nltk
nltk.download('stopwords')


def word_frequencies(list_sentences, title):
    list_sentences_frequency = {}
    tmp_dict = {'word_frequencies': '', 'value_word_in_sent': ''}
    # Each sentence
    for index, sentence in enumerate(list_sentences):
        sentence = list_sentences_to_string(sentence)
        word_frequencies = {}  # đếm mỗi từ xuất hiện bao nhiêu lần
        for word in nltk.word_tokenize(sentence):
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
        list_sentences_frequency[index] = word_frequencies

        word_in_sent = {}
        value_word_in_sent = {}
        n = len(list_sentences)
        for word in word_frequencies.keys():
            word_in_sent[word] = 0
            for sentence in list_sentences:
                if word in nltk.word_tokenize(sentence):
                    word_in_sent[word] += 1
            # trả về kiểu {'hi': 0.5, 'hello': 0.6,...}
            value_word_in_sent[word] = math.log(n/(1+word_in_sent[word]))

        tmp_dict['word_frequencies'] = word_frequencies
        tmp_dict['value_word_in_sent'] = value_word_in_sent
        list_sentences_frequency[index] = tmp_dict

    # All sentences
    list_sentences_string = list_sentences_to_string(list_sentences)
    word_frequencies = {}
    for word in nltk.word_tokenize(list_sentences_string):
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1

    word_in_sent = {}
    value_word_in_sent = {}
    n = len(list_sentences)
    for word in word_frequencies.keys():
        word_in_sent[word] = 0
        for sentence in list_sentences:
            if word in nltk.word_tokenize(sentence):
                word_in_sent[word] += 1
        value_word_in_sent[word] = math.log(n/(1+word_in_sent[word]))

    tmp_dict['word_frequencies'] = word_frequencies
    tmp_dict['value_word_in_sent'] = value_word_in_sent
    list_sentences_frequency['list_sentences'] = tmp_dict

    # Title
    title = list_sentences_to_string(title)
    word_frequencies = {}
    for word in nltk.word_tokenize(title):
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1

    word_in_sent = {}
    value_word_in_sent = {}
    n = len(list_sentences)
    for word in word_frequencies.keys():
        word_in_sent[word] = 0
        for sentence in list_sentences:
            if word in nltk.word_tokenize(sentence):
                word_in_sent[word] += 1
        value_word_in_sent[word] = math.log(n/(1+word_in_sent[word]))

    tmp_dict['word_frequencies'] = word_frequencies
    tmp_dict['value_word_in_sent'] = value_word_in_sent
    list_sentences_frequency['title'] = tmp_dict
    return list_sentences_frequency


def list_sentences_to_string(list_sentences):
    return " ".join(list_sentences)


def return_vocab(list_sentences_frequency, key='list_sentences'):
    return list(list_sentences_frequency[key]['word_frequencies'])


def weight(list_sentences_frequency, key):
    word_frequencies = list_sentences_frequency[key]['word_frequencies']
    value_word_in_sents = list_sentences_frequency[key]['value_word_in_sent']
    if len(word_frequencies) == 0:
        maximum_frequency = 0
    else:
        maximum_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        if maximum_frequency == 0:
            word_frequencies[word] = 0
        else:
            word_frequencies[word] = (
                word_frequencies[word]/maximum_frequency)*value_word_in_sents[word]
    return word_frequencies


def normalize(vec):
    return vec / np.sqrt(np.sum(vec ** 2))


def simCos(vec1, vec2):
    norm_vec1 = normalize(vec1)
    norm_vec2 = normalize(vec2)
    return np.sum(norm_vec1 * norm_vec2)

# S là dict của 1 câu, cần chuyển kích thước bằng vs dict của 1 document


def get_vec_weight(S, voca):
    vec = np.zeros(len(voca))
    for i in range(len(voca)):
        try:
            vec[i] = S[voca[i]]

        except:
            continue
    return vec


def sim_2_sent(list_sentences_frequency):
    voca = return_vocab(list_sentences_frequency)
    sim2sents = []
    for i in range(len(list_sentences_frequency)-2):
        sim2sents.append([])
        for j in range(len(list_sentences_frequency)-2):
            sent1_dict = weight(list_sentences_frequency, key=i)
            sent1_vec = get_vec_weight(sent1_dict, voca)
            sent2_dict = weight(list_sentences_frequency, key=j)
            sent2_vec = get_vec_weight(sent2_dict, voca)
            sim2sents[i].append(simCos(sent1_vec, sent2_vec))
    return sim2sents


def sim_with_title(list_sentences_frequency):
    voca = return_vocab(list_sentences_frequency)
    title_dict = weight(list_sentences_frequency, key='title')
    title_vec = get_vec_weight(title_dict, voca)
    simWithTitle = []
    # duyet tung sentence
    for i in range(len(list_sentences_frequency)-2):
        s_i_dict = weight(list_sentences_frequency, key=i)
        s_i = get_vec_weight(s_i_dict, voca)
        simT = simCos(s_i, title_vec)
        simWithTitle.append(simT)
    return simWithTitle


def sim_with_doc(list_sentences_frequency, index_sentence):
    voca = return_vocab(list_sentences_frequency)
    d_dict = weight(list_sentences_frequency, key='list_sentences')
    document_ = get_vec_weight(d_dict, voca)
    sentence_dict = weight(list_sentences_frequency, index_sentence)
    sentence_ = get_vec_weight(sentence_dict, voca)
    return simCos(sentence_, document_)


def count_noun(sentences):  # đếm số danh từ
    number_of_nouns = []
    for sentence in sentences:
        text = nltk.word_tokenize(sentence)
        post = nltk.pos_tag(text)
        # noun_list = ['NN', 'NNP', 'NNS', 'NNPS']
        noun_list = ['NNP']
        num = 0
        for k, v in post:
            if v in noun_list:
                #            if v.startswith('NN'):
                num += 1
        number_of_nouns.append(num)
    return number_of_nouns


def preprocess_raw_sent(raw_sent, tmp=False):
    words = nltk.word_tokenize(raw_sent)
    preprocess_words = ""
    stopwords = nltk.corpus.stopwords.words('english')
    # stemmer= PorterStemmer()
    for word in words:
        if word.isalpha():
            if word not in stopwords:
                if tmp == False:
                    word = word.lower()
                preprocess_words += " " + word
    preprocess_words = preprocess_words.strip()
    return preprocess_words


def preprocess_numberOfNNP(raw_sent):
    words = nltk.word_tokenize(raw_sent)
    preprocess_words = ""
    stopwords = nltk.corpus.stopwords.words('english')
    # stemmer= PorterStemmer()
    for word in words:
        if word.isalpha():
            if word not in stopwords:
                word = " " + word
                preprocess_words += word
    preprocess_words = preprocess_words.strip()
    return preprocess_words


def preprocess_for_article(raw_sent):
    words = nltk.word_tokenize(raw_sent)
    preprocess_words = ""
    # stopwords = nltk.corpus.stopwords.words('english')
    # stemmer= PorterStemmer()
    for word in words:
        # if word.isalpha():
        # if word not in stopwords:
        # word = word.lower()
        # word = stemmer.stem(word)
        # word =  + word" "
        preprocess_words += " " + word
    preprocess_words = preprocess_words.strip()
    return preprocess_words
