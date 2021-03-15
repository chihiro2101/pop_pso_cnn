import preprocess
import math
import nltk
import numpy as np


class METRIC(object):
    def __init__(self, title, sentences, agent, simWithTitle, simWithDoc, sim2sents, number_of_nouns, order_params):
        self.title = title
        self.sentences = sentences
        self.n = len(sentences)
        self.values = agent
        self.simWithTitle = simWithTitle
        self.simWithDoc = simWithDoc
        self.sim2sents = sim2sents
        self.number_of_nouns = number_of_nouns
        self.sum_nouns = sum(number_of_nouns)
        self.order_params = order_params

    # number of sentences in summary
    def O(self):
        return np.sum(self.values)

    def position(self):
        p = 0
        for i in range(self.n):
            if self.values[i] == 1:
                p = p + math.sqrt(1 / (i + 1))
        return p

    def scale_noun(self):
        scale = 0
        for i in range(self.n):
            if self.values[i] == 1:
                scale += self.number_of_nouns[i]
        return scale / self.sum_nouns

    def relationT(self):
        rt = 0
        for i in range(self.n):
            if self.values[i] == 1:
                rt += self.simWithTitle[i]/(self.O())
        return rt

    def cohesion(self):
        C = 0
        M_init = []
        for i in range(self.n - 1):
            if self.values[i] == 1:
                for j in range(i+1, self.n):
                    if self.values[j] == 1:
                        sim_2sents = self.sim2sents[i][j]
                        C = C + sim_2sents
                        M_init.append(sim_2sents)
        Ns = (self.O())*(self.O() - 1)/2.0

        Cs = C/Ns
        if len(M_init) == 0:
            M = 0
        else:
            M = max(M_init)
        CoH = (math.log(Cs*9.0+1.0))/(math.log(M*9.0+1.0) + 0.00001)
        return CoH

    def Cov(self):
        cov = 0
        for i in range(self.n - 1):
            if self.values[i] == 1:
                for j in range(i+1, len(self.sentences)):
                    if self.values[j] == 1:
                        cov += self.simWithDoc[i] + self.simWithDoc[j]
        return cov

    def leng(self):
        length = {}
        for i in range(self.n):
            if self.values[i] == 1:
                length[i] = self.words_count(self.sentences[i])

        length_value = list(length.values())
        std = np.std(length_value)
        avgl = np.mean(length_value)

        le = 0
        for i in range(len(length_value)):
            if std == 0:
                sigmoid = 0.98
            else:
                sigmoid = np.exp((-length_value[i] - avgl) / std)
            le += (1 - sigmoid) / (1 + sigmoid)
        return le

    def words_count(self, sentences):
        words = nltk.word_tokenize(sentences)
        return len(words)

    def fitness(self):
        # pos = 0.35
        # rel = 0.35
        # cover = 0.005
        # coh = 0.005
        # le  = 0.29
        if self.order_params == 0:
            # parameters set 0
            rel = 0.25
            le = 0.3
            pos = 0.15
            noun = 0.3
        elif self.order_params == 1:
            # parameters set 1
            rel = 0.3
            le = 0.3
            pos = 0.15
            noun = 0.25
        elif self.order_params == 2:
            # parameters set 2
            rel = 0.3
            le = 0.3
            pos = 0.1
            noun = 0.3
        # elif self.order_params == 3:
            # parameters set 3
        #     rel = 0.35
        #     le = 0.2
        #     pos = 0.2
        #     noun = 0.25
        # elif self.order_params == 4:
        #     # parameters set 4
        #     rel = 0.35
        #     le = 0.2
        #     pos = 0.1
        #     noun = 0.35
        # elif self.order_params == 5:
        #     rel = 0.25
        #     le = 0.3
        #     pos = 0.15
        #     noun = 0.3

        # # My
        # rel = 0.25
        # le  = 0.3
        # pos = 0.15
        # noun = 0.3

        fit = pos*self.position() + rel*self.relationT() + le * self.leng() + noun*self.scale_noun()
        # fit = pos*self.position() + rel*self.relationT() + cover*self.Cov() + coh*self.cohesion() +le*self.leng()
        return fit

    def GLS(self):
        sim_sent = []
        sim_sent = self.simWithTitle
        c = []
        d = []
        p = [0]*self.n
        max_sim = max(sim_sent)
        for i in range(self.n):
            c.append(math.sqrt(1 / (i + 1)) + sim_sent[i]/max_sim)
            d.append(c[i]/(1+p[i]))

        gls = 0
        for i in range(self.n):
            if d[i] == min(d):
                p[i] += 1
            gls += 0.5*p[i]*self.values[i]
        return self.fitness() - gls


def compute_fitness(title, sentences, agent, simWithTitle, simWithDoc, sim2sents, number_of_nouns, order_params):
    metric = METRIC(title, sentences, agent, simWithTitle,
                    simWithDoc, sim2sents, number_of_nouns, order_params)
    # return metric.GLS()
    return metric.fitness()
