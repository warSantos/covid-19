from sys import argv, exit
import os
import pandas as pd
import numpy as np

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

    return vertexes, edges, cities_df, cities_new_cases

def concat(parameters, sep):
    text = ''
    for i in range(len(parameters)):
        text += str(parameters[i])

        if i != len(parameters) - 1:
            text += sep
        else:
            text += '\n'

    return text