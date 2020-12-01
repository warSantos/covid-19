import pandas as pd
import igraph
from pprint import pprint

df = pd.read_csv('tiradentes.txt')

grafo = igraph.Graph(directed=True)

grafo.add_vertices(len(df))

i = 0
for row in df.itertuples():
    grafo.vs[i]['name'] = row.cod_ibge
    grafo.vs[i]['label'] = row.nome
    grafo.vs[i]['lati'] = row.lati
    grafo.vs[i]['long'] = row.long
    i += 1

arestas = []

for i in range(len(df)-1):
    arestas.append([i,len(df)-1])
    arestas.append([len(df)-1, i])

grafo.add_edges(arestas)

layout = grafo.layout_davidson_harel()
igraph.plot(grafo,"tiradentes.pdf", layout=layout, bbox = (800, 800))
grafo.save("tiradentes.gml", format="gml")

pt = open('grafo_tiradentes.txt','w')
total = len(grafo.vs)
for i in range(total):
    pt.write(str(i)+',')
    l = []
    for j in range(total):
        if grafo.get_eid(i, j, directed=True, error=False) > -1:
            l.append(str(j))
    pt.write(','.join(l)+'\n')
pt.close()