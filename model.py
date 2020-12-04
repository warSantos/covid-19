from sys import argv, exit
import sys
sys.path.append('src')
import os
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from graph import Graph
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
    
    # executa o projeção novamente com os pesos que ajustaram a curva melhor
    # para cada um dos set de parâmetros.
    max_pred = 0
    df_sets = pd.read_csv('dados/best_set.csv')
    for row in df_sets.itertuples():
        
        
        weights = np.array([ float(i) for i in row.weights.replace('[','').replace(']','').split() ])
        graph.setWeights(city, row.c, weights)
        
        predictions = np.zeros(shape=(30,len(cities_new_cases[city])-1))

        for i in range(1):
            graph.resetVertexValues()
            predictions[i] = graph.predict_cases(len(cities_new_cases[city])-1, city, debug=True)

        mean_prediction = predictions.mean(axis=0)
        if max_pred < mean_prediction[len(cities_new_cases[city])-2]:
            max_pred = mean_prediction[len(cities_new_cases[city])-2]
        
        plt.plot(mean_prediction)
        
        plt.xlabel("Qtde. Dias")
        plt.ylabel("Qtde. Casos")
    
    plt.plot(accumulated_curve, label="Real Curve")
    x = list(range(0,len(accumulated_curve),10))
    plt.xticks(x)
    y_min = 0
    y_max = max(accumulated_curve[-1], max_pred)
    plt.grid()
    #plt.vlines(n_steps, ymin=y_min, ymax=y_max, colors='red',  linestyles='dashed', label='train/test')
    plt.ylim([0,y_max])
    plt.legend()
    plt.savefig('plots/aprox_all.pdf')