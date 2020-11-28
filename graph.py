from sys import argv, exit
import os
import igraph
import pandas as pd
import numpy as np
import random

class Graph():

    def __init__(self, vertexes, edges, df, initial_values, initial_sum):

        # Criando um grafo direcionado.
        self.graph = igraph.Graph(directed=True)

        # casos ativos no início do treino (soma dos novos casos dos últimos 14 dias)
        self.initial_values = []
        self.initial_sum = initial_sum

        self.m = len(edges)
        self.n = len(vertexes)

        # taxa de recuperação diária do coronavírus
        self.c = random.random()
       
        self.graph.add_vertices(vertexes)
        self.graph.add_edges(edges)
        
        #define que o grafo é ponderado
        self.graph.es["weight"] = 1.0        

        for i in range(0, len(vertexes)):
            self.graph.vs[i]["label"] = df.iloc[i]['nome']
            self.graph.vs[i]["name"] = df.iloc[i]['cod_ibge']
            self.graph.vs[i]["lati"] = df.iloc[i]['lati']
            self.graph.vs[i]["long"] = df.iloc[i]['long']
            self.initial_values.append(initial_values[initial_values['ibgeID'] == df.iloc[i]['cod_ibge']].iloc[0]['newCases'])
            self.graph.vs[i]["value"] = self.initial_values[i]


        self.graph.save("grafos/grafo.gml", format="gml")

    def setWeights(self, c, weights):
        self.c = c

        i = 0
        for edge in self.graph.get_edgelist():

            self.graph[edge[0], edge[1]] = weights[i]
            i += 1
    
    def getTotalCases(self):
        sum = 0
        for v in self.graph.vs:
            sum += v['value']

        return sum

    def resetVertexValues(self):
        for i in range(self.n):
            self.graph.vs[i]["value"] = self.initial_values[i]

    def autoUpdateCases(self):
        newVertexesValue = np.zeros(shape=self.n)

        for i in range(self.n):
            sum = 0
            for j in range(0, self.n):
                sum += self.graph.vs[j]['value']*self.graph[j, i]

            newVertexesValue[i] = self.graph.vs[i]['value']*(1 - self.c) + sum

        for i in range(self.n):
            self.graph.vs[i]["value"] = newVertexesValue[i]

    def predict_cases(self, n_steps):
        # casos acumulados até o momento de início
        sum = self.initial_sum - np.sum(self.initial_values)
        cumulative_cases = []

        for i in range(n_steps):
            sum += self.getTotalCases()
            cumulative_cases.append(sum)
            
            self.autoUpdateCases()

        return cumulative_cases