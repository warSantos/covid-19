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
    
    return vertexes, edges, cities_df

if __name__=='__main__':

    vertexes, edges, cities_df = readFiles()

    # casos ativos de teste para o algoritimo.
    #initial_values = np.random.uniform(0, 5, size=len(vertexes))
    
    # curva de teste para o algoritimo.
    #real_curve = np.random.uniform(400, 4000, size=35)

    df_regiao = pd.read_csv('dados/df_micro.csv').sort_values(by=['epi_week'])
    # Iniciando o algoritimo a partir da 13 semanda de pandemia.
    # A primeira semana a ser contabilizada foi a 13 em SJ. A pandemia
    # só passou a ser contabilizada em todas as cidades a partir da
    # semana 26.
    start = 12
    d_cases = {}
    for row in df_regiao.query('epi_week > %s' % start)[['ibgeID','epi_week','totalCases']].itertuples():
        #print(row.city, row.epi_week, row.totalCases)
        # Se a semana ainda não existe no dicionário.
        if row.epi_week not in d_cases:
            d_cases[row.epi_week] = {}
        # Se a cidade não estiver associada a semana.
        if row.ibgeID not in d_cases[row.epi_week]:
            d_cases[row.epi_week][row.ibgeID] = row.totalCases
        # Verificando se esse foi o valor máximo de casos nessa semana para essa cidade.
        if row.totalCases > d_cases[row.epi_week][row.ibgeID]:
            d_cases[row.epi_week][row.ibgeID] = row.totalCases
    
    pprint(d_cases)
    # Gerando curva total de casos.
    real_curve = []
    for epi_week in d_cases:
        total_week = 0
        for city in d_cases[epi_week]:
            total_week += d_cases[epi_week][city]
        real_curve.append(total_week)
    
        
    # Construindo a configuração inicial de cada vértice.
    weeks = list(d_cases.keys())
    weeks.sort()
    # Selecionando a semana de configuração dos vértices.
    start = 13
    week_start = weeks[start]
    print("Semana de inicio: ", week_start)
    initial_values = []
    for row in cities_df.itertuples():
        # Se a cidade foi contabilizada nessa semana de pandemia.
        if row.cod_ibge in d_cases[week_start]:
            initial_values.append(d_cases[week_start][row.cod_ibge])
        # Se a cidade não foi contabilizada assuma que ela tem 0 casos.
        else:
            initial_values.append(0)
    
    real_curve = np.array(real_curve)
    initial_values = np.array(initial_values)

    # Repartindo a curva real em teste e treino.
    size_sample_train = 5
    print("Tamanho do conjunto de treino: ",size_sample_train)
    train = real_curve[:start+size_sample_train]
    test = real_curve[start+size_sample_train:]

    # Treinando o algoritimo.
    graph = Graph(vertexes, edges, cities_df, initial_values)
    ag = Ag(graph, train)

    # executa o algoritmo genético
    c, weights = ag.run(npop=35, nger=100, cp=0.9, mp=0.01, xmin=0.0, xmax=0.3)
    
    # executa o projeção novamente com os pesos que ajustaram a curva melhor
    graph.setWeights(c, weights)
    graph.resetVertexValues()
    prediction = graph.predict_cases(len(test))
    
    
    train_and_test = list(train)+list(prediction)
    plt.plot(train_and_test, label="Prediction")
    plt.plot(real_curve, label="Real Curve")
    y_min = real_curve[0]
    y_max = max(real_curve[-1], prediction[-1])
    plt.vlines(len(train)-1, ymin=y_min, ymax=y_max, colors='red',  linestyles='dashed', label='train/test')
    plt.legend()
    plt.show()