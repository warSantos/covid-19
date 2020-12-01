import pandas as pd
from sys import argv, exit
from pprint import pprint
import os
import matplotlib.pyplot as plt
import numpy as np

class Dataset():

    def __init__(self, caminho_dataset=None, dst='dados/df_micro.csv', base='grafos/cidades.txt'):

        if os.path.exists(dst):
            self.df_micro = pd.read_csv(dst)
            return

        # Carregando o dataset de cidades.
        df_cids = pd.read_csv(base)

        # Carregando o dataset de covid.
        df_covid = pd.read_csv(caminho_dataset)

        # Filtrando o dataset pelas cidades existentes na rede da micro região.
        self.df_micro = df_covid.query('ibgeID in %s' % list(set(df_cids['cod_ibge'])))
        
        print("Lista de cidades: ")
        pprint(set(self.df_micro['city']))
        self.df_micro.to_csv(dst)

    def distrib_regs(self, att, alvo, x_label, y_label, dst, figsize=None, pos=1):

        ## Distribuição de registros.
        cids = []
        for valor in sorted(set(self.df_micro[att])):
            temp = self.df_micro.query(att+" == '%s'" % valor)
            cids.append([temp.iloc[0][alvo], len(temp)])

        cids.sort(key=lambda x: x[pos])
        regs = [ cid[1] for cid in cids ]
        y_pos = np.arange(len(regs))
        labels = [ cid[0] for cid in cids ]
        if figsize is not None:
            plt.figure(figsize=figsize)
        plt.bar(y_pos,regs)
        plt.grid()
        plt.xticks(y_pos, labels, rotation=90)
        plt.ylabel(x_label)
        plt.ylabel(y_label)
        plt.subplots_adjust(bottom=0.5, top=0.99)
        plt.tight_layout()
        plt.plot()
        plt.savefig(dst)
    
    def distrib_casos(self, att='ibgeID'):

        ## Distribuição de casos.
        cids = {}
        for cid in sorted(set(self.df_micro[att])):
            temp = self.df_micro.query(att+" == '%s'" % cid)
            cids[cid] = {}
            # Verificando quantos casos a cidade tem por mês.
            for row in temp.itertuples():
                
                tks = row.date.split('-')
                tks.pop()
                mes = '-'.join(tks)
                
                mes = row.epi_week
                
                # Se este mês não estiver contabilidade no cidade.
                if mes not in cids[cid]:
                    cids[cid][mes] = 0
                cids[cid][mes] += row.newCases
        
        pprint(cids)

if __name__=='__main__':

    caminho_dataset = 'dados/cases-brazil-cities-time.csv'
    df = Dataset(caminho_dataset, argv[1], argv[2])
    # Dist. de registros por cidade.
    #df.distrib_regs('ibgeID','city', 'Qtde. Registros', 'Cidades', 'plots/dist_registros_cids.pdf')
    # Dist. de registros por data.
    #df.distrib_regs('date','date', 'Qtde. Registros', 'Datas', 'plots/dist_registros_datas.pdf', figsize=(26, 5), pos=0)
    # Dist. de casos por cidade.
    df.distrib_casos('city')