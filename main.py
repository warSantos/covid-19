from sys import argv, exit
import os
import igraph
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from ag import Ag
from graph import Graph


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

    # casos ativos iniciais de coronavírus em cada vértice. Definir posteriormente...
    initial_values = np.random.uniform(0, 100, size=len(vertexes))

    # essa é a curva que queremos ajustar o algoritmo
    real_curve = np.random.uniform(400, 4000, size=35)

    graph = Graph(vertexes, edges, cities_df, initial_values)
    ag = Ag(graph, real_curve)

    # executa o algoritmo genético
    c, weights = ag.run(npop=35, nger=100, cp=0.9, mp=0.01, xmin=0.0, xmax=0.3)
    
    # executa o projeção novamente com os pesos que ajustaram a curva melhor
    graph.setWeights(c, weights)
    graph.resetVertexValues()
    prediction = graph.predict_cases(len(real_curve))
    x = range(len(real_curve))
    
    plt.plot(x, prediction)
    plt.show()