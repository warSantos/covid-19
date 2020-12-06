from sys import argv, exit
import sys
sys.path.append('src')
import pandas as pd
import numpy as np
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
    #labels = ['First','Second','Third', 'Fourth', 'Fifth','Sixth','Seventh','Eighth','Ninth','Tenth']
    labels = ['Prediction']#,'Second','Third', 'Fourth', 'Fifth']
    label_count = 0
    plt.figure(figsize=(10,4))
    #for row in df_sets.head(10).itertuples():
    for row in df_sets.head(4).tail(1).itertuples():
        
        weights = np.array([ float(i) for i in row.weights.replace('[','').replace(']','').split() ])
        graph.setWeights(city, row.c, weights)
        
        predictions = np.zeros(shape=(30,len(cities_new_cases[city])-1))

        for i in range(30):
            graph.resetVertexValues()
            predictions[i] = graph.predict_cases(len(cities_new_cases[city])-1, city, debug=True)

        mean_prediction = predictions.mean(axis=0)

        with open('dados/pontos.txt', 'w') as fd:
            fd.write(','.join([ str(i) for i in mean_prediction ]))

        if max_pred < mean_prediction[len(cities_new_cases[city])-2]:
            max_pred = mean_prediction[len(cities_new_cases[city])-2]
        
        plt.plot(mean_prediction, label=labels[label_count], color="orange")
        
        plt.xlabel("Qtde. Dias")
        plt.ylabel("Qtde. Casos")
        label_count += 1
    
    plt.plot(accumulated_curve, label="Real Curve", color="blue", linestyle="dashed")
    x = list(range(0,len(accumulated_curve),10))
    plt.xticks(x)
    y_min = 0
    y_max = max(accumulated_curve[-1], max_pred)
    plt.grid()
    plt.vlines(n_steps, ymin=y_min, ymax=y_max, colors='red',  linestyles='dashed', label='train/test')
    plt.ylim([0,y_max])
    plt.xlim([0,156])
    plt.legend()
    plt.savefig('plots/aprox_all.pdf')