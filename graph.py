from sys import argv, exit
import os
import igraph
import pandas as pd

class Graph():

    def __init__(self):


        # Verificando se o grafo existe, se sim carregue ele.
        if os.path.exists('grafos/grafo.gml'):
            self.graph = igraph.load('grafos/grafo.gml')
            return
        
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

        # Criando um grafo direcionado.
        self.graph = igraph.Graph(directed=True)
        df = pd.read_csv('grafos/cidades.txt')
        
        nomes = list(set(df['nome']))
        nomes.sort()

        self.graph.add_vertices(vertexes)
        self.graph.add_edges(edges)
        for i in range(0, len(vertexes)):
            self.graph.vs[i]["label"] = nomes[i]

        #layout = self.graph.layout_kamada_kawai()
        #layout = self.graph.layout_grid()
        layout = self.graph.layout_davidson_harel(maxiter=1000)
        igraph.plot(self.graph,"grafos/grafo.pdf", layout=layout, bbox = (1100, 1000))
        self.graph.save("grafos/grafo.gml", format="gml")


if __name__=='__main__':

    Graph()