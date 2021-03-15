import random
from features import compute_fitness
from preprocess import preprocess_raw_sent
from preprocess import sim_with_title
from preprocess import sim_with_doc
from preprocess import sim_2_sent
from preprocess import count_noun
from preprocess import word_frequencies
from copy import copy
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import nltk
import os.path
import statistics as sta
import re
from preprocess import preprocess_for_article
from preprocess import preprocess_numberOfNNP
import time
import os
import glob
from rouge import Rouge
from shutil import copyfile
import pandas as pd
import math
import multiprocessing
import shutil

class Summerizer(object):
    def __init__(self, title, sentences, raw_sentences, population_size, max_generation, crossover_rate, mutation_rate, num_picked_sents, simWithTitle, simWithDoc, sim2sents, number_of_nouns, order_params, MinLT, MaxLT, scheme):
        self.title = title
        self.raw_sentences = raw_sentences
        self.sentences = sentences
        self.num_objects = len(sentences)
        self.population_size = population_size
        self.max_generation = max_generation
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.num_picked_sents = num_picked_sents
        self.simWithTitle = simWithTitle
        self.simWithDoc = simWithDoc
        self.sim2sents = sim2sents
        self.number_of_nouns = number_of_nouns
        self.order_params = order_params
        self.MaxLT = MaxLT
        self.MinLT = MinLT
        self.scheme = scheme


    def generate_population(self, amount):
        # print("Generating population...")
        population = []
        for i in range(amount):

            #position
            agent = np.zeros(self.num_objects)
            agent[np.random.choice(list(range(self.num_objects)), self.num_picked_sents, replace=False)] = 1
            agent = agent.tolist()
            
            #p_best_position
            pbest_position = agent

            #p_best_value
            fitness = compute_fitness(self.title, self.sentences, agent, self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns, self.order_params)
            
            #velocity
            velocity = np.zeros(self.num_objects)
            velocity[np.random.choice(list(range(self.num_objects)), self.num_picked_sents, replace=False)] = 1
            velocity = velocity.tolist()

            life_time = 0
            age = 0
            

            # print("fitness: {:.4f}" , format(fitness))
            # print(agent)
            population.append((agent, fitness, pbest_position, velocity, life_time, age))
        return population 


    def words_count(self, sentences):
        words = nltk.word_tokenize(sentences)
        return len(words)


    def sum_of_words(self, individual):
        sum = 0
        agent = individual[0][:]
        for i in range(self.num_objects):
            if agent[i]==1:
                sum += self.words_count(self.sentences[i])
        return sum


    def roulette_select(self, total_fitness, population):
        fitness_slice = np.random.rand() * total_fitness
        fitness_so_far = 0.0
        for phenotype in population:
            fitness_so_far += phenotype[1]
            if fitness_so_far >= fitness_slice:
                return phenotype
        return None


    def rank_select(self, population):
        ps = len(population)
        if ps == 0:
            return None
        population = sorted(population, key=lambda x: x[1], reverse=True)
        fitness_value = []
        for individual in population:
            fitness_value.append(individual[1])

        fittest_individual = max(fitness_value)
        medium_individual = sta.median(fitness_value)
        selective_pressure = fittest_individual - medium_individual
        j_value = 1
        a_value = np.random.rand()   
        for agent in population:
            if ps == 0:
                return None
            elif ps == 1:
                return agent
            else:
                range_value = selective_pressure - (2*(selective_pressure - 1)*(j_value - 1))/( ps - 1) 
                prb = range_value/ps
                if prb > a_value:
                    return agent
            j_value +=1

                
    def crossover(self, individual_1, individual_2, max_sent):
        # check tỷ lệ crossover
        if self.num_objects < 2 or random.random() >= self.crossover_rate:
            return individual_1[:], individual_2[:]
        
        #tìm điểm chéo 1
        crossover_point = 1 + random.randint(0, self.num_objects - 2)
        agent_1a = individual_1[0][:crossover_point] + individual_2[0][crossover_point:]
        fitness_1a = compute_fitness(self.title, self.sentences, agent_1a, self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns, self.order_params)
        agent_1b = individual_2[0][:crossover_point] + individual_1[0][crossover_point:]
        fitness_1b = compute_fitness(self.title, self.sentences, agent_1b, self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns, self.order_params)

        velocity = np.zeros(self.num_objects)
        velocity = velocity.tolist()

        if fitness_1a > fitness_1b:
            child_1 = (agent_1a, fitness_1a, agent_1a, velocity, 0, 0)
        else:
            child_1 = (agent_1b, fitness_1b, agent_1a, velocity, 0, 0)

        sum_sent_in_summary = sum(child_1[0])
        agent_1 = child_1[0]
        fitness_1 = child_1[1]
        if sum_sent_in_summary > max_sent:
            while(sum_sent_in_summary > max_sent):
                remove_point = 1 + random.randint(0, self.num_objects - 2)
                if agent_1[remove_point] == 1:
                    agent_1[remove_point] = 0
                    sum_sent_in_summary -=1            
            fitness_1 = compute_fitness(self.title, self.sentences, agent_1, self.simWithTitle, self.simWithDoc,self.sim2sents, self.number_of_nouns, self.order_params)
            child_1 = (agent_1, fitness_1, agent_1, velocity, 0, 0)


        #tìm điểm chéo 2
        crossover_point_2 = 1 + random.randint(0, self.num_objects - 2)
        
        agent_2a = individual_1[0][:crossover_point_2] + individual_2[0][crossover_point_2:]
        fitness_2a = compute_fitness(self.title, self.sentences, agent_2a, self.simWithTitle, self.simWithDoc,self.sim2sents, self.number_of_nouns, self.order_params)
        agent_2b = individual_2[0][:crossover_point_2] + individual_1[0][crossover_point_2:]
        # agent_2 = individual_2[0][:crossover_point] + individual_1[0][crossover_point:]
        fitness_2b = compute_fitness(self.title, self.sentences, agent_2b, self.simWithTitle, self.simWithDoc,self.sim2sents, self.number_of_nouns, self.order_params)
        if fitness_2a > fitness_2b:
            child_2 = (agent_2a, fitness_2a, agent_2a, velocity, 0, 0)
        else:
            child_2 = (agent_2b, fitness_2b, agent_2b, velocity, 0, 0)        
        
        sum_sent_in_summary_2 = sum(child_2[0])
        agent_2 = child_2[0]
        fitness_2 = child_2[1]        
        if sum_sent_in_summary_2 > max_sent:
            while(sum_sent_in_summary_2 > max_sent):
                remove_point = 1 + random.randint(0, self.num_objects - 2)
                if agent_2[remove_point] == 1:
                    agent_2[remove_point] = 0
                    sum_sent_in_summary_2 -= 1
            fitness_2 = compute_fitness(self.title, self.sentences, agent_2, self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns, self.order_params)
            child_2 = (agent_2, fitness_2, agent_2, velocity, 0, 0)
        return child_1, child_2
    

    def mutate(self, individual, max_sent):
        velocity_vector = individual[3]
        pbest_position = individual[2]
        sum_sent_in_summary = sum(individual[0])
        agent = individual[0][:]
        for i in range(len(agent)):
            if random.random() < self.mutation_rate and sum_sent_in_summary < max_sent :
                if agent[i] == 0 :
                   agent[i] = 1
                   sum_sent_in_summary +=1
                # else :
                #    agent[i] = 0
                #    sum_sent_in_summary -=1        
        fitness = compute_fitness(self.title, self.sentences, agent, self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns, self.order_params)
        return (agent, fitness, pbest_position, velocity_vector, 0, 0)

    def compare(self, lst1, lst2):
        for i in range(self.num_objects):
            if lst1[i] != lst2[i]:
                return False
        return True


    def calculate_lifetime(self, fitness, avg_fitness, max_fitness, min_fitness, scheme):
        eta = 1/2*(self.MaxLT - self.MinLT)

        if scheme == 0:
            life_time = min(self.MinLT + int(eta*(fitness/avg_fitness)), self.MaxLT)
        elif scheme == 1:
            life_time = self.MinLT + int(2*eta*(fitness - min_fitness)/(max_fitness - min_fitness))
        else:
            if fitness <= avg_fitness:
                life_time = self.MinLT + int(eta*(fitness - min_fitness)/(avg_fitness - min_fitness))
            else:
                life_time = int(0.5*(self.MinLT + self.MaxLT) + eta*(fitness - avg_fitness)/(max_fitness - avg_fitness))
        return life_time


    def evaluate_age(self, population, scheme):

        fitness_value = []
        for individual in population:
            fitness_value.append(individual[1])
        try:
            avg_fitness = sta.mean(fitness_value)
            max_fitness = max(fitness_value)
            min_fitness = min(fitness_value)
        except: 
            print("bug")
            import pdb; pdb.set_trace()


        #life_time
        new_population = []
        for individual in population:
            indiv = list(individual)
            indiv[4] =  self.calculate_lifetime(indiv[1], avg_fitness, max_fitness, min_fitness, scheme)
            indiv[5] += 1
            new_population.append(tuple(indiv))
            
        return new_population 


    def check_timelife(self, population):
        count = 0
        population = sorted(population, key=lambda x: x[1], reverse=True)
        chosen = int(0.1*len(population))
        new_population = population[ : chosen]
        for individual in population [chosen : ]:
            if individual[5] >= individual[4]:
                count +=1
            else:
                new_population.append(individual)

        return count, new_population

    def selection(self, population, popsize):
        if len(population) == 0:
            population = self.generate_population(self.population_size)
        population = self.evaluate_age(population, self.scheme)

        max_sent = int(0.3*len(self.sentences))
        if len(self.sentences) < 4:
            max_sent = len(self.sentences)       
        new_population = []

        population = sorted(population, key=lambda x: x[1], reverse=True)

        chosen_agents = int(0.1*len(population))
        
        elitism = population[: chosen_agents]
        population = population[chosen_agents : ]
        

        total_fitness = 0
        for indivi in population:
            total_fitness = total_fitness + indivi[1]  

        population_size = popsize
        cpop = 0.0
        check_time_global = time.time()
        while cpop <= population_size or (time.time() - check_time_global) > 300:
            population = sorted(population, key=lambda x: x[1], reverse=True)
            parent_1 = None

            check_time_1 = time.time()
            while parent_1 == None:
                parent_1 = self.rank_select(population)
                if parent_1 == None and (time.time() - check_time_1) > 60:
                    try:
                        parent_1 = random.choice(population)
                    except:
                        return self.generate_population(population_size), self.population_size
         
            parent_2 = None
            check_time_2 = time.time()
            while parent_2 == None :
                parent_2 = self.roulette_select(total_fitness, population)
                if parent_2 == None and (time.time() - check_time_2) > 60:
                    try:
                        parent_2 =  random.choice(population)
                    except:
                        return self.generate_population(population_size), self.population_size
                
                if parent_2 != None:
                    if self.compare(parent_2[0], parent_1[0]) :
                        parent_2 = self.roulette_select(total_fitness, population)
            parent_1, parent_2 = copy(parent_1), copy(parent_2)
            child_1, child_2 = self.crossover(parent_1, parent_2, max_sent)

            # child_1
            individual_X = self.mutate(child_1, max_sent)
            check1 = 0
            check2 = 0
            if len(population) > 4 :
                competing = random.sample(population, 4)
                lowest_individual = min(competing , key = lambda x: x[1])
                if individual_X[1] > lowest_individual[1]:
                    new_population.append(individual_X)
                    check1 = 1
                elif sum(lowest_individual[0]) <= max_sent:
                    new_population.append(lowest_individual)
                    check1 = 1

            # child_2
            individual_Y = self.mutate(child_2, max_sent)
            if len(population) > 4 :
                competing_2 = random.sample(population, 4)
                lowest_individual_2 = min(competing_2 , key = lambda x: x[1])
                if individual_Y[1] > lowest_individual_2[1]:
                    new_population.append(individual_Y)
                    check2 = 1
                elif sum(lowest_individual_2[0]) <= max_sent:
                    new_population.append(lowest_individual_2)
                    check2 = 1
            if check1 + check2 == 0:
                cpop += 0.1
            else:
                cpop += check1 + check2


        new_size = len(new_population)
        if new_size != 0:
            new_population = self.evaluate_age(new_population, self.scheme)
        elif new_size == 0 and len(population) != 0:
            return elitism.extend(population), population_size
        else:
            new_population = self.generate_population(population_size)
            new_population = self.evaluate_age(new_population, self.scheme)
        Dsize, current_population = self.check_timelife(population)
        new_population.extend(elitism)
        new_population.extend(current_population)
        new_popsize = popsize + new_size - Dsize
        if new_size > 60:
            new_size = 60

        fitness_value = []

        for individual in new_population:
            fitness_value.append(individual[1])

        try:
            avg_fitness = sta.mean(fitness_value)
        except:
            return self.generate_population(population_size), self.population_size


        agents_in_Ev = [] 
        for agent in new_population:
            if (agent[1] > 0.95*avg_fitness) and (agent[1] < 1.05*avg_fitness):
                agents_in_Ev.append(agent)

        if len(agents_in_Ev) >= len(new_population)*0.9 :
            new_population = self.generate_population(int(0.7*self.population_size)) 
            agents_in_Ev = sorted(agents_in_Ev, key=lambda x: x[1], reverse=True)
            chosen = self.population_size - len(new_population)
            new_population.extend(agents_in_Ev[: chosen])
            # for x in agents_in_Ev:
            #     new_population.append(x)
            #     if len (new_population) == self.population_size:
            #         break
        return new_population, new_popsize

    def normalize(self, chromosome):
        for i in range(len(chromosome)):
            if chromosome[i] < 0.5 :
                chromosome[i] = 0
            else:
                chromosome[i] = 1
        return chromosome
    
    def subtraction(self, bin_arr1, bin_arr2 ):
        ans = np.zeros(self.num_objects)
        for i in range(len(bin_arr1)):
            if bin_arr1[i] == 0 and bin_arr2[i] == 0:
                ans[i] = 0
            elif bin_arr1[i] == 1 and bin_arr2[i] == 1:
                ans[i] = 0
            else:
                ans[i] = 1
        return ans

    def reduce_mem(self, ans, max_sent):

        sum_sent_in_summary = sum(ans)
        if sum_sent_in_summary > max_sent:
            while(sum_sent_in_summary > max_sent):
                for remove_point in range(self.num_objects - 1, 0, -1):
                    if ans[remove_point] == 1:
                        ans[remove_point] = 0
                        sum_sent_in_summary -=1     
        return ans  

    def solveGA(self, population):
        popsize = self.population_size
        for i in range(self.max_generation):
            population, popsize = self.selection(population, popsize)
        return population

    def PSO(self):

        W = 0.5
        c1 = 0.5
        c2 = 0.9
        n_iterations = 30 

        max_sent = int(0.3*len(self.sentences))
        if len(self.sentences) < 4:
            max_sent = len(self.sentences) 


        gbest_position = np.zeros(self.num_objects)
        gbest_position[np.random.choice(list(range(self.num_objects)), self.num_picked_sents, replace=False)] = 1
        gbest_position = gbest_position.tolist()
        gbest_fitness_value = compute_fitness(self.title, self.sentences, gbest_position, self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns, self.order_params)
        

        #init population
        population = self.generate_population(self.population_size)
        
        for i in tqdm(range(n_iterations)):
            for i, individual in enumerate(population):
                individual = list(individual)
                fitness_candidate = compute_fitness(self.title, self.sentences, individual[0], self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns, self.order_params)
                if fitness_candidate > individual[1]:
                    individual[1] = fitness_candidate 
                    individual[2] = individual[0] #pbest of individual
                if fitness_candidate > gbest_fitness_value:
                    gbest_fitness_value = fitness_candidate
                    gbest_position = individual[0]
                population[i] = tuple(individual)

            for i, individual in enumerate(population):
                individual = list(individual)
                particle_position_vector = np.array(individual[0])
                pbest_position = np.array(individual[2]) 
                velocity_vector = np.array(individual[3])
                gbest = np.array(gbest_position)

                # new_velocity = (W*velocity_vector) + (c1*random.random())*(pbest_position - particle_position_vector) + (c2*random.random())*(gbest - particle_position_vector)
                new_velocity = (W*velocity_vector) + (c1*random.random())*self.subtraction(pbest_position , particle_position_vector) + (c2*random.random())*self.subtraction(gbest , particle_position_vector)
                new_velocity = new_velocity.tolist()
                individual[3] = self.normalize(new_velocity)
                new_velocity = np.array(individual[3])
                particle_position_vector = self.subtraction(particle_position_vector, new_velocity)
                individual[0] = self.reduce_mem(particle_position_vector.tolist(), max_sent)
                population[i] = tuple(individual)

            populationGA = self.solveGA(population)
            populationGA = sorted(populationGA, key=lambda x: x[1], reverse=True)
            populationPSO = sorted(population, key=lambda x: x[1], reverse=True)
            combine =  int(self.population_size/2)
            population = populationPSO[: combine ]
            for individual in populationGA[ : combine] :
                population.append(individual)

                
        return self.find_best_individual(population)

              
    def find_best_individual(self, population):
        if len(population) == 0:
            return None
        best_individual = sorted(population, key=lambda x: x[1], reverse=True)[0]
        return best_individual
 

    
    def show(self, individual,  file):
        index = individual[2]
        f = open(file,'w', encoding='utf-8')
        for i in range(len(index)):
            if index[i] == 1:
                f.write(self.raw_sentences[i] + '\n')
        f.close()

def load_a_doc(filename):
    file = open(filename, encoding='utf-8')
    article_text = file.read()
    file.close()
    return article_text   


def load_docs(directory):
    docs = list()
    list_file = os.listdir(directory)
    random.shuffle(list_file)
    for name in list_file:
        filename = directory + '/' + name
        doc = load_a_doc(filename)
        docs.append((doc, name))
    return docs

def clean_text(text):
    cleaned = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", ",")).strip()
    check_text = "".join((item for item in cleaned if not item.isdigit())).strip()
    if len(check_text.split(" ")) < 4:
        return 'None'
    return cleaned

def start_run(processID, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, sub_stories, save_path, order_params, scheme):
   
    for example in sub_stories:
        start_time = time.time()
        # raw_sentences = re.split("\n\s+", example[0])
        raw_sents = re.split("\n", example[0])
        df = pd.DataFrame(raw_sents, columns =['raw'])
        df['preprocess_raw'] = df['raw'].apply(lambda x: clean_text(x))
        newdf = df.loc[(df['preprocess_raw'] != 'None')]
        raw_sentences = newdf['preprocess_raw'].values.tolist()
        if len(raw_sentences) == 0:
            continue
       
        # print('raw', len(raw_sentences), stories.index(example))
        title_raw = raw_sentences[0]
        # Preprocessing
        # print("Preprocessing...")
        sentences = []
        sentences_for_NNP = []
        for raw_sent in raw_sentences:
            sent = preprocess_raw_sent(raw_sent)
            # sent_tmp = preprocess_numberOfNNP(raw_sent)
            
            sent_tmp = preprocess_raw_sent(raw_sent, True)
            if len(sent.split(' ')) < 2:
                raw_sentences.remove(raw_sent)
            else:
                sentences.append(sent)
                sentences_for_NNP.append(sent_tmp)
        title = preprocess_raw_sent(title_raw)
        list_sentences_frequencies = word_frequencies(sentences, title)
        number_of_nouns = count_noun(sentences_for_NNP)
        simWithTitle = sim_with_title(list_sentences_frequencies)
        sim2sents = sim_2_sent(list_sentences_frequencies)
        simWithDoc = []
        # for sent in sentences:
        for i in range(len(sentences)):
            simWithDoc.append(sim_with_doc(list_sentences_frequencies, index_sentence=i))

          
        print("Done preprocessing!")
        # DONE!
        print('time for processing', time.time() - start_time)
        if len(sentences) < 4:
            NUM_PICKED_SENTS = len(sentences)
        else:
        #            NUM_PICKED_SENTS = x
        #       NUM_PICKED_SENTS = int(len(sentences)*0.2)
            NUM_PICKED_SENTS = 4

        MinLT = 1
        MaxLT = 7

        Solver = Summerizer(title, sentences, raw_sentences, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, NUM_PICKED_SENTS, simWithTitle, simWithDoc, sim2sents, number_of_nouns, order_params, MinLT, MaxLT, scheme)
        best_individual = Solver.PSO()
        file_name = os.path.join(save_path, example[1])         

        if best_individual is None:
            print ('No solution.')
        else:
            print(file_name)
            print(best_individual)
            Solver.show(best_individual, file_name)
        
def a_process_do(processID, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, sub_stories, save_path, next_part):
        rouge_score = []
        for i in range(3): 
            #chạy từng bộ
            start_run(processID, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, sub_stories, save_path, 0, i) 
            #tính rouge của từng bộ
            rouge1, rouge2, rougeL = evaluate_rouge(save_path)
            rouge_score.append((i,rouge1, rouge2, rougeL))
        # weights_had_max_value = max(rouge_score, key = lambda i : i[1])[0]
        scheme_had_max_value = max(rouge_score, key = lambda i : i[1])[0]
        result_file = '{}.{}'.format(processID, 'txt')
        fp = open(result_file,'w', encoding='utf-8')
        fp.write('\n'.join('{} , {} , {} , {} '.format(x[0],x[1], x[2], x[3]) for x in rouge_score))
       
        # chay weights do tren 1 tap ngau nhien
        # random_part = random.choice(set_of_docs)
        # while random_part == sub_stories:
        #     random_part = random.choice(set_of_docs)
        # print("best_weights_for_a_loop: " , weights_had_max_value )
        save_path_for_valid = 'hyp'
        start_run(processID, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, next_part, save_path_for_valid, 0, scheme_had_max_value)
       

def multiprocess(num_process, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, stories, save_path):
    processes = []

    # num_docs_per_loop = math.floor(len(stories)/5)
    n = 100
    set_of_docs = [stories[i:i + n] for i in range(0, len(stories), n)] 

    # import pdb
    # pdb.set_trace()

    # for i in range(num_process):
    for index, sub_stories in enumerate(set_of_docs):
        if index == 5:
            p = multiprocessing.Process(target=a_process_do, args=(
                index, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE,sub_stories, save_path[-1], set_of_docs[0]))
            processes.append(p)
            p.start()
        else:
            p = multiprocessing.Process(target=a_process_do, args=(
                index, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE,sub_stories, save_path[index], set_of_docs[index+1]))
            processes.append(p)
            p.start()
        
    for p in processes:
        p.join()


def evaluate_rouge(hyp_path):
    hyp = hyp_path
    raw_ref = 'duc2002_summaries_I'
    FJoin = os.path.join
    files_hyp = [FJoin(hyp, f) for f in os.listdir(hyp)]
    files_raw_ref = [FJoin(raw_ref, f) for f in os.listdir(hyp)]
    
    f_hyp = []
    f_raw_ref = []
    print("number of document: ", len(files_hyp))
    for file in files_hyp:
        f = open(file)
        f_hyp.append(f.read())
        f.close()
    for file in files_raw_ref:
        f = open(file)
        f_raw_ref.append(f.read())
        f.close()
        
    rouge_1_tmp = []
    rouge_2_tmp = []
    rouge_L_tmp = []
    for hyp, ref in zip(f_hyp, f_raw_ref):
        try:
            rouge = Rouge()
            scores = rouge.get_scores(hyp, ref, avg=True)
            rouge_1 = scores["rouge-1"]["r"]
            rouge_2 = scores["rouge-2"]["r"]
            rouge_L = scores["rouge-l"]["r"]
            rouge_1_tmp.append(rouge_1)
            rouge_2_tmp.append(rouge_2)
            rouge_L_tmp.append(rouge_L)
        except Exception:
            pass


    rouge_1_avg = sta.mean(rouge_1_tmp)
    rouge_2_avg = sta.mean(rouge_2_tmp)
    rouge_L_avg = sta.mean(rouge_L_tmp)
    print('Rouge-1: ', rouge_1_avg)
    print('Rouge-2: ',rouge_2_avg )
    print('Rouge-L: ', rouge_L_avg)

    for path in os.listdir(hyp_path):
        full_path = os.path.join(hyp_path, path)
        os.remove(full_path)

    return rouge_1_avg, rouge_2_avg, rouge_L_avg        
            


def main():
    # Setting Variables
    POPU_SIZE = 30
    MAX_GEN = 4
    CROSS_RATE = 0.8
    MUTATE_RATE = 0.4
    #NUM_PICKED_SENTS = 4

    directory = 'duc2002_documents_2'
    save_path=['hyp1', 'hyp2', 'hyp3', 'hyp4', 'hyp5', 'hyp6']

    if not os.path.exists('hyp1'):
        os.makedirs('hyp1')
    if not os.path.exists('hyp2'):
        os.makedirs('hyp2')
    if not os.path.exists('hyp3'):
        os.makedirs('hyp3')
    if not os.path.exists('hyp4'):
        os.makedirs('hyp4')
    if not os.path.exists('hyp5'):
        os.makedirs('hyp5')
    if not os.path.exists('hyp6'):
        os.makedirs('hyp6')




    print("Setting: ")
    print("POPULATION SIZE: {}".format(POPU_SIZE))
    print("MAX NUMBER OF GENERATIONS: {}".format(MAX_GEN))
    print("CROSSING RATE: {}".format(CROSS_RATE))
    print("MUTATION SIZE: {}".format(MUTATE_RATE))

    # list of documents
    stories = load_docs(directory)
    start_time = time.time()

    
    multiprocess(6, POPU_SIZE, MAX_GEN, CROSS_RATE,
                 MUTATE_RATE, stories, save_path)
    # start_run(1, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, stories, save_path[0], 0, 0)

    print("--- %s mins ---" % ((time.time() - start_time)/(60.0*len(stories))))

if __name__ == '__main__':
    main()              
     
        
        
     
    


    
    
    
    
        
            
            
         
