import random
import time
import numpy as np
import json
import matplotlib as plt
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

def runGetCorrectPattern(correct_pattern, nbits):
    correct = False
    numberOfAttempts = 0
    start = time.time()
    while not correct:
        random_pattern = np.random.randint(0,2 ,nbits)

        comparison = random_pattern == correct_pattern
        numberOfAttempts += 1
        if(comparison.all()):
            correct = True
            end = time.time()
    


    return {
        'timeTaken':end - start,
        'numberOfAttempts':numberOfAttempts
    }

if(__name__ == '__main__'):
    testResults = {}
    n_bits = [2, 4, 8, 12, 16, 24, 32]
    number_of_tests = 30
    random.seed(123123)
    for test in range(0, number_of_tests):
        for nbit in n_bits:
            correct_pattern = np.random.randint(0,2 ,nbit)
            paterndata = runGetCorrectPattern(correct_pattern, nbit)
            print(paterndata)

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


    fig1, ax = plt.subplots()
    ax.set_title('Agent execution time (to reach the final goal) per episode')
    bp = ax.boxplot(runTimes)
    data = open('results.json',)
    average_attempts = [] 
    data = json.load(data)
    for item in data:
        for res in data['results']:
            average_attempts 
            plt.show()

    # print(random_pattern)