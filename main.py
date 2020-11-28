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

    week = 7

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
    # Identificando o maior número de casos em cada dia (pode aumentar mais de uma vez em um dia).
    for row in df_regiao.query('epi_week > %s' % start)[['ibgeID','date','totalCases']].itertuples():
        # Se a data ainda não existe no dicionário.
        if row.date not in d_cases:
            d_cases[row.date] = {}
        # Se a cidade não estiver associada a data.
        if row.ibgeID not in d_cases[row.date]:
            d_cases[row.date][row.ibgeID] = row.totalCases
        # Verificando se esse foi o valor máximo de casos nessa data para essa cidade.
        if row.totalCases > d_cases[row.date][row.ibgeID]:
            d_cases[row.date][row.ibgeID] = row.totalCases
    
    pprint(d_cases)
    # Gerando curva total de casos.
    real_curve = []
    for date in d_cases:
        total_day = 0
        for city in d_cases[date]:
            total_day += d_cases[date][city]
        real_curve.append(total_day)
    
        
    # Construindo a configuração inicial de cada vértice.
    days = list(d_cases.keys())
    days.sort()
    
    # Selecionando a semana de configuração dos vértices.
    start_week = 13
    start = week * start_week
    day_start = days[start]
    print("Total de dias: ", len(d_cases.keys()))
    print("Dia de inicio: ", start)
    print("Semana de inicio: ", start_week)
    initial_values = []
    # Configurando o setup inicial dos vértices.
    for row in cities_df.itertuples():
        # Se a cidade foi contabilizada nessa semana de pandemia.
        if row.cod_ibge in d_cases[day_start]:
            initial_values.append(d_cases[day_start][row.cod_ibge])
        # Se a cidade não foi contabilizada assuma que ela tem 0 casos.
        else:
            initial_values.append(0)
    
    real_curve = np.array(real_curve)
    initial_values = np.array(initial_values)

    # Repartindo a curva real em teste e treino.
    size_sample_train = week * 5
    print("Qtde dias de treino: ",size_sample_train)
    print("Intervalo de treino: ", start, " - ", start+size_sample_train)
    print("Curva Real: ", real_curve)
    train = real_curve[start:start+size_sample_train]
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
    
    
    train_and_pred = list(real_curve[:start+size_sample_train])+list(prediction)
    plt.plot(train_and_pred, label="Prediction")
    plt.plot(real_curve, label="Real Curve")
    plt.grid()
    plt.xlabel("Qtde. Dias")
    plt.ylabel("Qtde. Casos")
    plt.xticks(list(range(0,len(real_curve),10)))
    y_min = real_curve[0]
    y_max = max(real_curve[-1], prediction[0])
    plt.vlines(start, ymin=y_min, ymax=y_max, colors='black',  linestyles='dotted', label='Start Point')
    plt.vlines(start+size_sample_train-1, ymin=y_min, ymax=y_max, colors='red',  linestyles='dashed', label='train/test')
    plt.ylim([0,real_curve[-1]])
    plt.legend()
    plt.savefig('plots/aprox.pdf')