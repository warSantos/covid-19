from sys import argv, exit
import os
import igraph
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt


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

class Graph():

    def __init__(self, vertexes, edges, df, initial_values):

        # Criando um grafo direcionado.
        self.graph = igraph.Graph(directed=True)

        self.m = len(edges)
        self.n = len(vertexes)

        # taxa de recuperação diária do coronavírus
        self.c = random.random()
        print('c:', self.c)
       
        self.graph.add_vertices(vertexes)
        self.graph.add_edges(edges)
        
        #define que o grafo é ponderado
        self.graph.es["weight"] = 1.0        

        for i in range(0, len(vertexes)):
            self.graph.vs[i]["label"] = df.iloc[i]['nome']
            self.graph.vs[i]["name"] = df.iloc[i]['cod_ibge']
            self.graph.vs[i]["lati"] = df.iloc[i]['lati']
            self.graph.vs[i]["long"] = df.iloc[i]['long']
            self.graph.vs[i]["value"] = initial_values[i]

        self.graph.save("grafos/grafo.gml", format="gml")

    def setWeights(self, weights):
        i = 0
        for edge in self.graph.get_edgelist():

            self.graph[edge[0], edge[1]] = weights[i]
            i += 1
    
    def getTotalCases(self):
        sum = 0
        for v in self.graph.vs:
            sum += v['value']

        return sum

    def autoUpdateCases(self):

        newVertexesValue = np.zeros(shape=self.n)

        for i in range(self.n):
            sum = 0
            for j in range(0, self.n):
                sum += self.graph.vs[j]['value']*self.graph[j, i]

            newVertexesValue[i] = self.graph.vs[i]['value']*(1 - self.c) + sum

        for i in range(self.n):
            self.graph.vs[i]["value"] = newVertexesValue[i]

if __name__=='__main__':

    vertexes, edges, cities_df = readFiles()

    # casos ativos de coronavírus em cada vértice. Definir posteriormente...
    initial_values = np.random.uniform(0, 100, size=len(vertexes))

    graph = Graph(vertexes, edges, cities_df, initial_values)
    
    # pesos a serem definidos pelo algoritmo genético
    weights = np.random.uniform(0, 0.2, size=graph.m)

    graph.setWeights(weights)


    x = range(10)
    y = []
    for i in range(10):
        
        y.append(graph.getTotalCases())
        print(y[i])
        
        #print(graph.getTotalCases())
        graph.autoUpdateCases()
    
    plt.plot(x, y)
    plt.show()