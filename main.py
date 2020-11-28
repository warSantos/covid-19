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


week_days = 7

def readFiles():
    
    vertexes = []
    edges = []
    graph_file = 'grafos/grafo.txt'
    with open(graph_file, 'r') as fd:
        # Para cada lista de edges de cada vértice.
        for l in fd:
            tokens = l.replace('\n','').split(',')
            vertex = tokens.pop(0)
            vertexes.append(vertex)
            # Criando lista de arestas para cada vértice.
            for v in tokens:
                edges.append([vertex, v])

    cities_df = pd.read_csv('grafos/cidades.txt')

    df_region = pd.read_csv('dados/df_micro.csv').sort_values(by=['epi_week'])
    # A primeira semana a ser contabilizada foi a 13 em SJ. A pandemia
    # só passou a ser contabilizada em todas as cidades a partir da
    # semana 26.

    # pegar os novos casos da semana 28-2 por cada cidade para
    initial_values = df_region.query('epi_week > 25 and epi_week < 27')[['ibgeID','newCases']].groupby(['ibgeID']).sum().reset_index()

    df_real_curve = df_region.query('epi_week > 27')[['date','totalCases']].groupby(['date']).sum().reset_index()
    df_real_curve = df_real_curve.sort_values(by=['date'], ascending=True)
    real_curve = []
    for row in df_real_curve.itertuples():
        real_curve.append(row.totalCases)
    
    initial_sum = real_curve[0]
    
    real_curve = np.array(real_curve)
    
    return vertexes, edges, cities_df, real_curve, initial_values, initial_sum

if __name__=='__main__':

    vertexes, edges, cities_df, real_curve, initial_values, initial_sum = readFiles()


    # Repartindo a curva real em teste e treino.
    size_sample_train = 63
    print("Curva Real: ", real_curve)
    train = real_curve[:size_sample_train]
    test = real_curve[size_sample_train:]

    # Treinando o algoritimo.
    graph = Graph(vertexes, edges, cities_df, initial_values, initial_sum)
    ag = Ag(graph, train)

    # executa o algoritmo genético
    c, weights = ag.run(npop=50, nger=150, cp=0.9, mp=0.01, xmin=0.0, xmax=0.3)

    print(c, weights)
    
    # executa o projeção novamente com os pesos que ajustaram a curva melhor
    graph.setWeights(c, weights)
    graph.resetVertexValues()
    prediction = graph.predict_cases(len(real_curve))
    
    
    #train_and_pred = list(train)+list(prediction)
    #plt.plot(train_and_pred, label="Prediction")
    plt.plot(prediction, label="Prediction")
    plt.plot(real_curve, label="Real Curve")
    plt.grid()
    plt.xlabel("Qtde. Dias")
    plt.ylabel("Qtde. Casos")
    plt.xticks(list(range(0,len(real_curve),10)))
    y_min = 0
    y_max = max(real_curve[-1], prediction[0])
    plt.vlines(size_sample_train-1, ymin=y_min, ymax=y_max, colors='red',  linestyles='dashed', label='train/test')
    plt.ylim([0,real_curve[-1]])
    plt.legend()
    plt.savefig('plots/aprox.pdf')