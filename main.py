from sys import argv, exit
import os
import igraph
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from ag import Ag
from graph import Graph
from pprint import pprint
import multiprocessing as mp
import time


def readFiles():

    vertexes = []
    edges = []
    graph_file = 'grafos/grafo_tiradentes.txt'
    with open(graph_file, 'r') as fd:
        # Para cada lista de edges de cada vértice.
        for l in fd:
            tokens = l.replace('\n', '').split(',')
            vertex = tokens.pop(0)
            vertexes.append(vertex)
            # Criando lista de arestas para cada vértice.
            for v in tokens:
                edges.append([vertex, v])

    cities_df = pd.read_csv('grafos/tiradentes.txt')

    df_region = pd.read_csv('dados/df_tiradentes.csv').sort_values(
        by=['epi_week'])
    # A primeira semana a ser contabilizada foi a 13 em SJ. A pandemia
    # só passou a ser contabilizada em todas as cidades a partir da
    # semana 26.

    df_new_cases = df_region.query('epi_week >= 27')

    cities_new_cases = {}

    # Para cada cidade.
    for cityID in set(df_new_cases['ibgeID']):
        temp = df_new_cases.query("ibgeID == '%s'" % cityID).sort_values(
            by=['date'], ascending=True)['newCases']
        cities_new_cases[cityID] = np.array(temp)
    """
        x = list(range(len(cities_new_cases[cityID])))
        plt.plot(x, cities_new_cases[cityID], label=cityID)
    plt.legend()
    plt.savefig("plots/curvesTR.pdf")
    plt.clf()
    """

    return vertexes, edges, cities_df, cities_new_cases


def run(self, graph, accumulated_curve, city, cities_new_cases):

    ag = Ag(graph, accumulated_curve[0:n_steps], city)

    # executa o algoritmo genético
    c, weights = ag.run(npop=30,
                        nger=100,
                        cp=0.9,
                        mp=0.01,
                        xmaxc=2.0,
                        xmax_edge=80)

    # executa o projeção novamente com os pesos que ajustaram a curva melhor
    graph.setWeights(city, c, weights)

    predictions = np.zeros(shape=(30, len(cities_new_cases[city]) - 1))

    for i in range(30):
        graph.resetVertexValues()
        predictions[i] = graph.predict_cases(len(cities_new_cases[city]) - 1,
                                             city,
                                             debug=True)

    mean_prediction = predictions.mean(axis=0)

    plt.plot(mean_prediction, label="Prediction")

    plt.plot(accumulated_curve, label="Real Curve")
    plt.grid()
    plt.xlabel("Qtde. Dias")
    plt.ylabel("Qtde. Casos")
    plt.xticks(list(range(0, len(accumulated_curve), 10)))
    y_min = 0
    y_max = max(accumulated_curve[-1],
                mean_prediction[len(cities_new_cases[city]) - 2])

    plt.vlines(n_steps,
               ymin=y_min,
               ymax=y_max,
               colors='red',
               linestyles='dashed',
               label='train/test')
    plt.ylim([0, y_max])
    plt.legend()
    plt.savefig('plots/aprox.pdf')


if __name__ == '__main__':

    vertexes, edges, cities_df, cities_new_cases = readFiles()

    graph = Graph(vertexes, edges, cities_df, cities_new_cases)

    n_steps = 105
    city = 3168804  # Tiradentes
    accumulated_curve = []

    for i in range(len(cities_new_cases[city])):

        if i == 0:
            accumulated_curve.append(cities_new_cases[city][0])
        else:
            accumulated_curve.append(cities_new_cases[city][i] +
                                     accumulated_curve[i - 1])

    processes = list()
    max_p = 30
    execs_count = 0

    ### Executando os processos em paralelo. ###

    # Enquanto o todos os testes não forem executados.
    while execs_count < n_tests:
        # Identifique os processos que estão vivos.
        processes = list(filter(lambda x: x.is_alive(), processes))
        # Se a quantidade de processos vivos for igual a quantidade
        # de processos permitidos rodar ao mesmo tempo.
        if len(processes) == max_p:
            # Espere cinco segundos para verificar novamente.
            time.sleep(5)
        else:
            # Se não crie mais um processo.
            p = mp.Process(name=str(execs_count),
                           target=run,
                           args=( # Argumentos para o método.
                               graph,
                               accumulated_curve,
                               city,
                               cities_new_cases,
                           ))
            processes.append(p)
            p.start()
            execs_count += 1

    for processo in processes:
        processo.join()
