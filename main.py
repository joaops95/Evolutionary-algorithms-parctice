import random
import time
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.setrecursionlimit(10**7) # max depth of recursion

# Master mind
# criar um padrao random de 0 e 1s com um numero de bits

# mostrar resultados boxplot de execution time e boxplot de reward ou avalicao


#2 criar uma funcao para mudar um bit dando um padrao
# paramos se nao encontrarmos duma solucao pmelhor apos mil mutacoes
# so aceitamos se forem melhores

#3 gerar padroes random uma populacao avaliar escolher os melhores etc

#4 crossover vs mutacao e combinar crossover e mutacao

#5 acertar nao so nas cores mas tambem na quantidade de bits

#6 padrao nao e so binario, podemos ter mais cores

#7 como poderiamos utilizar um algoritmo evolucionario
def evaluationForPatter(eval_correct_pattern, eval_random_pattern):
    return (abs(len(eval_correct_pattern) - np.sum(eval_correct_pattern == eval_random_pattern)))
    # raise Exception

def fitnessForPatter(eval_correct_pattern, eval_random_pattern):
    return np.sum(eval_correct_pattern == eval_random_pattern)
    # raise Exception



def mutateBit(muttated_random_pattern):
    random_index = np.random.randint(0,len(muttated_random_pattern))

    if(muttated_random_pattern[random_index] == 0):
        muttated_random_pattern[random_index] = 1
    else: 
        muttated_random_pattern[random_index] = 0

    return muttated_random_pattern
    # raise Exception

def applyPopulationOnDataset(dataset, nbits, function_apply):
    for item in dataset.to_numpy():
        random_population = []
        for i in range(0, 100):
            random_population.append(function_apply(list(item)))

        df = pd.DataFrame(np.asarray(random_population))

        df['fitness'] = df.apply(lambda x: fitnessForPatter(x, correct_pattern), axis=1)
        df.sort_values(by='fitness', ascending=False, inplace=True)
        print(df['fitness'].iloc[0])
        if(df['fitness'].iloc[0] == nbit):
            raise Exception
        df = df.iloc[:int(len(df)*0.3)]
        df.drop(['fitness'], axis=1, inplace=True)

        applyPopulationOnDataset(df, nbits, mutateBit)


def runGetCorrectPattern(correct_pattern, nbits, population = False):
    correct = False
    numberOfAttempts = 0
    start = time.time()
    random_pattern = np.random.randint(0,2 ,nbits)

    while not correct:
        print(random_pattern)
        comparison = random_pattern == correct_pattern
        better_pattern_found = False
        idx = 0
        while (idx < 1000 and not comparison.all()):
            idx += 1
            if(population):
                    random_population = np.random.randint(0,2, size=(100, nbits))
                    df = pd.DataFrame(random_population)
                    df['fitness'] = df.apply(lambda x: fitnessForPatter(x, correct_pattern), axis=1)
                    df.sort_values(by='fitness', ascending=False, inplace=True)
                    # print(df['fitness'].iloc[0])
                    # print(len(df))
                    df = df.iloc[:int(len(df)*0.3)]
                    df.drop(['fitness'], axis=1, inplace=True)
                    value = applyPopulationOnDataset(df, nbits, mutateBit)
                    if(df['fitness'].iloc[0] == nbit):
                        print("found")
                        raise Exception
            else:
                random_copy = random_pattern.copy()
                muttated_bit = mutateBit(random_copy)

                fitness_mutated = fitnessForPatter(correct_pattern,muttated_bit)
                fitness_random = fitnessForPatter(correct_pattern,random_pattern)
                print(f"{fitness_mutated} - {fitness_random}")
                if(fitness_mutated > fitness_random):
                    random_pattern = muttated_bit.copy()
            # print(evaluationForPatter(correct_pattern, random_pattern))
        numberOfAttempts += 1
        if(comparison.all()):
            correct = True
            end = time.time()
    


    return {
        'timeTaken':end - start,
        'numberOfAttempts':numberOfAttempts
    }

def index_in_list(a_list, index):
    return index < len(a_list)


if(__name__ == '__main__'):
    testResults = {}
    showResults = []
    n_bits = [1024, 12, 16, 18, 22, 24, 64, 128, 256, 512, 1024, 1500, 2048, 2512]
    number_of_tests = 30
    # random.seed(123123) #used in exercise 1
    random.seed(time.time()) #used in exercise 1
    
    for test in range(0, number_of_tests):
        for nbit in n_bits:
            correct_pattern = np.random.randint(0,2 ,nbit)
            paterndata = runGetCorrectPattern(correct_pattern, nbit, True)

            try:
                testResults[test]['results'].append({
                    'nbits': nbit,
                    'execution_time':paterndata['timeTaken'],
                    'attempts':paterndata['numberOfAttempts']

                })
            except KeyError:
                testResults[test] = {}
                testResults[test]['results'] = [
                    {
                    'nbits': nbit,
                    'execution_time':paterndata['timeTaken'],
                    'attempts':paterndata['numberOfAttempts']
                    }
                ]
            with open('results.json', 'w') as outfile:
                json.dump(testResults, outfile, indent=4, sort_keys=True)



    data = open('results.json',)
    attempts = [] 
    timestamps = [] 
    positions = []

    data = json.load(data)
    for item in data.keys():
        item = data[item]

        for i in range(0, len(item['results'])):
            if(not index_in_list(attempts, i)):
                attempts.append([])
            if(not index_in_list(timestamps, i)):
                timestamps.append([])
            print(i)
            attempts[i].append(item['results'][i]['attempts'])
            timestamps[i].append(item['results'][i]['execution_time'])
            if(not item['results'][i]['nbits'] in positions):
                positions.append(item['results'][i]['nbits'])



    fig1, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax.set_title('Box plot number of attempts throught the bit increasing')
    ax2.set_title('Box plot execution times throught the bit increasing')

    # print(attempts)
    # print(timestamps)

    bp = ax.boxplot(attempts)
    bp = ax2.boxplot(timestamps)
    ax.set_xticklabels(positions)
    ax2.set_xticklabels(positions)

    plt.show()

# , positions=[2,4,5.5]
    # print(random_pattern)