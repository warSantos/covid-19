from sys import argv, exit
import os
import igraph
import pandas as pd
import numpy as np
import random
from datetime import datetime
from random import random

class Graph():

    def __init__(self, vertexes, edges, df, initial_values, initial_sum, initial_sum_cities):

        # Criando um grafo direcionado.
        self.graph = igraph.Graph(directed=True)

        # casos ativos no início do treino (soma dos novos casos dos últimos 14 dias)
        self.initial_values = []
        self.initial_sum = initial_sum
        self.initial_sum_cities = initial_sum_cities

        self.m = len(edges)
        self.n = len(vertexes)
      
        self.graph.add_vertices(vertexes)
        self.graph.add_edges(edges)
        
        #define que o grafo é ponderado
        self.graph.es["weight"] = 1.0        

        for i in range(0, len(vertexes)):
            self.graph.vs[i]["label"] = df.iloc[i]['nome']
            self.graph.vs[i]["name"] = df.iloc[i]['cod_ibge']
            self.graph.vs[i]["lati"] = df.iloc[i]['lati']
            self.graph.vs[i]["long"] = df.iloc[i]['long']
            self.initial_values.append(initial_values[initial_values['ibgeID'] == df.iloc[i]['cod_ibge']].iloc[0]['newCases']/7)
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
                if self.graph.vs[j]["value"] > 0 and self.graph[j, i] > random():
                    sum += 1#round(self.graph.vs[j]['value']*self.graph[j, i])
            newVertexesValue[i] = round(self.graph.vs[i]['value'] * self.c[i]) + sum

        for i in range(self.n):
            self.graph.vs[i]["value"] = newVertexesValue[i]

    def predict_cases(self, n_steps, debug=False):
        
        def concat(parameters, sep):
            text = ''
            for i in range(len(parameters)):
                text += str(parameters[i])

                if i != len(parameters) -1:
                    text += sep
                else:
                    text +='\n'

            return text
        
        def printVertexes(step, fileOut, vertexes):
            for vertex in vertexes:
                fileOut.write(concat([step, vertex['name'], vertex['label'], vertex['value']], ','))
        

        # casos acumulados até o momento de início
        
        sum = self.initial_sum - np.sum(self.initial_values)
        cumulative_cases_group = []

        cumulative_cases_cities = {}
        for v in self.graph.vs:
            cumulative_cases_cities[v['name']] = np.zeros(n_steps)

        if debug:
            fileOut = open('logs/' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '.csv', 'w')
            fileOut.write(concat(['step','ibge','name','newCases'], ','))
       
        for i in range(n_steps):

            sum += self.getTotalCases()
            cumulative_cases_group.append(sum)

            if debug:
                printVertexes(i, fileOut, self.graph.vs)

            self.autoUpdateCases()
            
            # Para cada cidade (vértice).
            for v in self.graph.vs:
                sum_city = 0
                if i == 0:
                    sum_city = self.initial_sum_cities[v['name']] - v['value']
                else:
                    sum_city = cumulative_cases_cities[v['name']][i-1]
                
                cumulative_cases_cities[v['name']][i] = sum_city + v['value']

        if debug:
            fileOut.close()

        return cumulative_cases_group, cumulative_cases_cities