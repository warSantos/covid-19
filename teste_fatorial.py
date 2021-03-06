from sys import argv, exit
import sys
sys.path.append('src')
import os
import igraph
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from ag import Ag
from graph import Graph
from pprint import pprint
import multiprocessing as multip
import time
from utils import readFiles


def run(graph, accumulated_curve, city, cities_new_cases, nger, npop, cp, mp, xmax_edge, id_exec, process_id):

    file = open('logs/logsProcess'+str(process_id)+'.csv', 'w')
    ag = Ag(graph, accumulated_curve[0:n_steps], city)

    # executa o algoritmo genético
    c, weights = ag.run(npop,
                        nger,
                        cp,
                        mp,
                        2.0,
                        xmax_edge,
                        fileOut=file,
                        id_exec=id_exec)

    file.close()


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
    max_p = 35
    
    options_nger = [70, 100, 150]   # opções número de gerações
    options_npop = [20, 30, 50]     # opções tamanho população
    options_maxedge = [70, 90, 110] # opções peso máximo aresta
    options_mp = [0.05, 0.01]       # opções probabilidade de mutação
    options_cp = [0.8, 1.0]         # opções probabilidade de cruzamento

   
    ### Executando um teste fatorial em paralelo. ###
    process_id = 0
    for nger in options_nger:
        for npop in options_npop:
            for max_edge in options_maxedge:
                for mp in options_mp:
                    for cp in options_cp:
                        for id_exec in range(10):

                            # Identifique os processos que estão vivos.
                            processes = list(filter(lambda x: x.is_alive(), processes))
                        
                            # Se a quantidade de processos vivos for igual a quantidade
                            # de processos permitidos rodar ao mesmo tempo.
                        
                            while(len(processes) == max_p):
                                # Espere cinco segundos para verificar novamente.
                                time.sleep(5)
                                processes = list(filter(lambda x: x.is_alive(), processes))

                            p = multip.Process(
                                name=str(process_id),
                                target=run,
                                args=(  # Argumentos para o método.
                                    graph,
                                    accumulated_curve,
                                    city,
                                    cities_new_cases,
                                    nger,
                                    npop,
                                    cp,
                                    mp,
                                    max_edge,
                                    id_exec,
                                    process_id
                                ))
                            processes.append(p)
                            p.start()
                            
                            process_id += 1
                            print(str(process_id) + " processos despachados")

    for processo in processes:
        processo.join()
