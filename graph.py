from sys import argv, exit
import os
import igraph
import pandas as pd
import numpy as np
import random
from datetime import datetime
from random import random


class Graph():

    def __init__(self, vertexes, edges, df, cities_new_cases):

        # Criando um grafo direcionado.
        self.graph = igraph.Graph(directed=True)

        self.cities_new_cases = cities_new_cases

        self.m = len(edges)
        self.n = len(vertexes)

        self.graph.add_vertices(vertexes)
        self.graph.add_edges(edges)

        # define que o grafo é ponderado
        self.graph.es["weight"] = 1.0

        self.dict_ibge_index = {}

        for i in range(0, len(vertexes)):
            self.graph.vs[i]["name"] = df.iloc[i]['nome']
            self.graph.vs[i]["id"] = df.iloc[i]['cod_ibge']
            self.graph.vs[i]["lati"] = df.iloc[i]['lati']
            self.graph.vs[i]["long"] = df.iloc[i]['long']
            self.graph.vs[i]["value"] = cities_new_cases[df.iloc[i]
                                                         ['cod_ibge']][0]
            self.dict_ibge_index[df.iloc[i]['cod_ibge']] = i

        # self.graph.save("grafos/grafo.gml", format="gml")

    def setWeights(self, ibge_id, c, weights):
        # descobre o índice no grafo de uma determinada cidade
        index_current_vertex = self.dict_ibge_index[ibge_id]

        self.c = c

        # altera apenas as arestas que incidem em um determinado vértice
        i = 0
        for edge in self.graph.get_edgelist():
            if edge[1] == index_current_vertex:
                self.graph[edge[0], edge[1]] = weights[i]
                i += 1

            if i == len(weights):
                break

    # soma todos os novos casos de todas as cidades no passo corrente
    def getTotalNewCases(self):
        sum = 0
        for v in self.graph.vs:
            sum += v['value']

        return sum

    # reseta o valor dos vértices para o valor inicial passado em cities_new_cases
    def resetVertexValues(self):
        for i in range(self.n):
            self.graph.vs[i]["value"] = self.cities_new_cases[self.graph.vs[i]['id']][0]

    # avança um passo na predição

    def autoUpdateCases(self, step, ibge_id):
        newVertexesValue = np.zeros(shape=self.n)
        index_current_vertex = self.dict_ibge_index[ibge_id]

        # se o vértice for = ibge_id então realiza a previsão baseada nos vizinhos
        # se não, apenas atualiza os valores com base nos dados já conhecidos

        for i in range(self.n):
            if i == index_current_vertex:
                sum = 0
                for j in range(0, self.n):
                    if self.graph[j, i] > 0:
                        if random() < self.graph.vs[j]["value"]/self.graph[j, i]:
                            sum += 1

                newVertexesValue[i] = self.graph.vs[i]['value'] * self.c + sum

            else:
                newVertexesValue[i] = self.cities_new_cases[self.graph.vs[i]["id"]][step]

        for i in range(self.n):
            self.graph.vs[i]["value"] = newVertexesValue[i]

    # atualiza os novos casos n passos para a cidade ibge_id e retorna a curva acumulada

    def predict_cases(self, n_steps, ibge_id, debug=False):

        def concat(parameters, sep):
            text = ''
            for i in range(len(parameters)):
                text += str(parameters[i])

                if i != len(parameters) - 1:
                    text += sep
                else:
                    text += '\n'

            return text

        def printVertexes(step, fileOut, vertexes):
            for vertex in vertexes:
                fileOut.write(
                    concat([step, vertex['id'], vertex['name'], vertex['value']], ','))

        if debug:
            fileOut = open(
                'logs/' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '.csv', 'w')
            fileOut.write(concat(['step', 'ibge', 'name', 'newCases'], ','))

        cumulative_cases_city = []

        sum = 0
        for i in range(n_steps):

            sum += self.graph.vs[self.dict_ibge_index[ibge_id]]['value']
            cumulative_cases_city.append(sum)

            if debug:
                printVertexes(i, fileOut, self.graph.vs)

            self.autoUpdateCases(i+1, ibge_id)

        if debug:
            fileOut.close()

        return cumulative_cases_city
