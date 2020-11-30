import math
from os import error
import numpy as np
import random
from graph import Graph

class Ag():

    def __init__(self, graph, real_curve, cities_curves):
        self.graph = graph
        self.real_curve = real_curve
        self.n_steps_prediction = len(real_curve)
        self.cities_curves = cities_curves

    def sum_residual_squares(self, vector1, vector2):
        sum = 0
        for i in range(self.n_steps_prediction):
            sum += pow(vector1[i] - vector2[i], 2)

        return sum

    def fitness_function(self, x):
        fits_x = []

        self.graph.setWeights(x[0:self.graph.n], x[self.graph.n:])

        for i in range(8):

            self.graph.resetVertexValues()

            prediction, cities_pred = self.graph.predict_cases(self.n_steps_prediction)
            
            #return self.sum_residual_squares(prediction, self.real_curve)

            # Para cada cidade (Ajustando pela curva de cada cidade).
            total = 0
            for cityID in self.cities_curves:
                total += self.sum_residual_squares(cities_pred[cityID], self.cities_curves[cityID])
            
            fits_x.append(math.sqrt(total))

        return np.mean(fits_x)

    def evaluate_pop(self):
        for i in range(self.npop):
            self.fit[i] = self.fitness_function(self.pop[i])

        # adequações para o problema de minimização pelo complemento
        sum = np.sum(self.fit)
        max = np.max(self.fit)

        sum = self.npop*max - sum

        for i in range(self.npop):
            self.normalized_fit[i] = (max - self.fit[i])/sum

    def roulette(self):
        r = random.random()
        sum = 0

        for i in range(self.npop):
            sum += self.normalized_fit[i]
            
            if r <= sum or i == self.npop-1:
                return i

    def parents_selection(self):
        
        i = 0
        while(i < self.npop):
            
            # caso de npop ser ímpar
            if (i+1 == self.npop):
                self.parents[i] = self.roulette()

            else:
                p1 = self.roulette()
                p2 = self.roulette()
                while (p1 == p2):
                    p2 = self.roulette()

                self.parents[i] = p1
                self.parents[i+1] = p2
        
            i += 2

    def cross(self, x_i, y_i, alpha, beta):
        def min(a, b):
            if a < b:
                return a
            return b
        
        def max(a, b):
            if a > b:
                return a
            return b

        if self.fit[y_i] < self.fit[x_i]:
            x = self.pop[y_i]
            y = self.pop[x_i]
        else:
            x = self.pop[x_i]
            y = self.pop[y_i]
        
        children = np.zeros(shape=(2, len(x)))

        for i in range(len(x)):
            xmax = self.xmax_edge
            if i < self.graph.n:
                xmax = self.xmaxc

            if x[i] <= y[i]:
                d = y[i] - x[i]
                children[0][i] = random.uniform(max(x[i] - alpha*d, self.xmin), min(y[i] + beta*d, xmax))
                children[1][i] = random.uniform(max(x[i] - alpha*d, self.xmin), min(y[i] + beta*d, xmax))

            else:
                d = x[i] - y[i]
                children[0][i] = random.uniform(max(y[i] - beta*d, self.xmin), min(x[i] + alpha*d, xmax))
                children[1][i] = random.uniform(max(y[i] - beta*d, self.xmin), min(x[i] + alpha*d, xmax))
            
        return children

    def crossover(self, alpha, beta):
        i = 0
        while(i < self.npop):
            
            if (i+1 == self.npop): # caso de npop ser ímpar
                self.itermediate_pop[i] = self.pop[self.parents[i]].copy()

            else:
                r = random.random()
                if(r <= self.cp):
                    children = self.cross(self.parents[i], self.parents[i+1], alpha, beta)
                    self.itermediate_pop[i] = children[0].copy()
                    self.itermediate_pop[i+1] = children[1].copy()
                else:
                    self.itermediate_pop[i] = self.pop[self.parents[i]].copy()
                    self.itermediate_pop[i+1] = self.pop[self.parents[i+1]].copy()

            i+=2

    def mutation(self):
        for i in range(self.npop):
            for j in range(len(self.itermediate_pop[i])):
                r = random.random()
                if(r <= self.mp):
                    xmax = self.xmax_edge
                    if j < self.graph.n:
                        xmax = self.xmaxc
                    self.itermediate_pop[i][j] = random.uniform(self.xmin, xmax)


    def run(self, npop, nger, cp, mp, xmaxc, xmax_edge):
        pop_c = np.random.uniform(0, xmaxc, size=(npop, self.graph.n))
        pop_edge = np.random.uniform(0, xmax_edge, size=(npop, self.graph.m))
        
        self.pop = np.concatenate((pop_c, pop_edge), axis=1)
        self.itermediate_pop = self.pop.copy()

        self.npop = npop  # número população
        self.cp = cp      # probabilidade de cruzamento
        self.mp = mp      # probabilidade de mutação
        self.xmin = 0
        self.xmaxc = xmaxc
        self.xmax_edge = xmax_edge

        self.fit = np.zeros(npop)
        self.normalized_fit = np.zeros(npop)

        self.parents = np.zeros(npop, dtype=int)

        self.evaluate_pop()

        elitism = True
        

        for g in range(1, nger):

            self.parents_selection()
            self.crossover(0.75, 0.25)

            self.mutation()

            best = self.pop[np.argmin(self.fit)].copy()
            self.pop = self.itermediate_pop.copy()
            
            self.evaluate_pop()
            
            if elitism:
                i_worst_individual = np.argmax(self.fit)
                self.pop[i_worst_individual] = best.copy()
                erro = self.fitness_function(best)
                self.fit[i_worst_individual] = erro
            
            print(np.min(self.fit))

        best = self.pop[np.argmin(self.fit)].copy()

        # retorna c e o peso das arestas
        return best[0:self.graph.n], best[self.graph.n:]