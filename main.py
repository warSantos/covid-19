from sys import argv, exit
import sys
sys.path.append('src')
import os
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from ag import Ag
from graph import Graph
from pprint import pprint
from utils import readFiles


if __name__=='__main__':

    vertexes, edges, cities_df, cities_new_cases = readFiles()
    
    # Treinando o algoritimo.
    graph = Graph(vertexes, edges, cities_df, cities_new_cases)
    
    n_steps=105
    city = 3168804  # Tiradentes
    accumulated_curve = []

    for i in range(len(cities_new_cases[city])):
        if i == 0:
            accumulated_curve.append(cities_new_cases[city][0])
        else:
            accumulated_curve.append( cities_new_cases[city][i] + accumulated_curve[i-1] )
    
    ag = Ag(graph, accumulated_curve[0:n_steps], city)

    # executa o algoritmo genético
    c, weights = ag.run(npop=30, nger=150, cp=1.0, mp=0.01, xmaxc=2.0, xmax_edge=100)

    print(c, weights)
    
    # executa o projeção novamente com os pesos que ajustaram a curva melhor
    graph.setWeights(city, c, weights)
    
    predictions = np.zeros(shape=(30,len(cities_new_cases[city])-1))

    for i in range(30):
        graph.resetVertexValues()
        predictions[i] = graph.predict_cases(len(cities_new_cases[city])-1, city, debug=True)

    mean_prediction = predictions.mean(axis=0)
    
    plt.plot(mean_prediction, label="Prediction")   
    

    plt.plot(accumulated_curve, label="Real Curve")
    plt.grid()
    plt.xlabel("Qtde. Dias")
    plt.ylabel("Qtde. Casos")
    plt.xticks(list(range(0,len(accumulated_curve),10)))
    y_min = 0
    y_max = max(accumulated_curve[-1], mean_prediction[len(cities_new_cases[city])-2])

    plt.vlines(n_steps, ymin=y_min, ymax=y_max, colors='red',  linestyles='dashed', label='train/test')
    plt.ylim([0,y_max])
    plt.legend()
    plt.savefig('plots/aprox.pdf')