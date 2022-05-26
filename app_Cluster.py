from datetime            import datetime
from PIL              import Image
from io                  import BytesIO

import pandas            as pd
import streamlit         as st
import numpy             as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


# Set no tema do seaborn para melhorar o visual dos plots
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)


# Função para ler os dados
@st.cache(show_spinner= True, allow_output_mutation=True)
def load_data(file_data):
    try:
        return pd.read_csv(file_data, sep=';')
    except:
        return pd.read_excel(file_data)

# Função para converter o df para csv
@st.cache
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Função para converter o df para excel
@st.cache
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def main():
    # Configuração inicial da página da aplicação
    st.set_page_config(page_title = 'Clusterização', \
        page_icon = 'img/cartao_icon.jpg',
        layout="wide",
        initial_sidebar_state='expanded'
    )

    # Título principal da aplicação
    st.write('# Clusterização - Cartão de Crédito')
    st.markdown("---")

    # Apresenta a imagem na barra lateral da aplicação
    image = Image.open("img/cartao.jpg")
    st.sidebar.image(image)

    # Botão para carregar arquivo na aplicação
    st.sidebar.write("## Suba o arquivo")
    data_file_1 = st.sidebar.file_uploader("C GENERAL", type = ['csv','xlsx'])

    # Verifica se há conteúdo carregado na aplicação
    if (data_file_1 is not None):
        df_raw = load_data(data_file_1)
        df = df_raw.copy()

        st.write('Visualizando o DataFrame')
        st.write(df.head())

        #tratando valores nulos
        df.loc[(df['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].median()
        df.loc[(df['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] = df['CREDIT_LIMIT'].median()

        #tratando outliers
        columns=['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT',
        'PAYMENTS', 'MINIMUM_PAYMENTS']

        for c in columns:
            
            Range=c+'_RANGE'
            df[Range]=0        
            df.loc[((df[c]>0)&(df[c]<=500)),Range]=1
            df.loc[((df[c]>500)&(df[c]<=1000)),Range]=2
            df.loc[((df[c]>1000)&(df[c]<=3000)),Range]=3
            df.loc[((df[c]>1000)&(df[c]<=5000)),Range]=4
            df.loc[((df[c]>5000)&(df[c]<=10000)),Range]=5
            df.loc[((df[c]>10000)),Range]=6

        columns=['BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 
        'CASH_ADVANCE_FREQUENCY', 'PRC_FULL_PAYMENT']

        for c in columns:
            
            Range=c+'_RANGE'
            df[Range]=0
            df.loc[((df[c]>0)&(df[c]<=0.1)),Range]=1
            df.loc[((df[c]>0.1)&(df[c]<=0.2)),Range]=2
            df.loc[((df[c]>0.2)&(df[c]<=0.3)),Range]=3
            df.loc[((df[c]>0.3)&(df[c]<=0.4)),Range]=4
            df.loc[((df[c]>0.4)&(df[c]<=0.5)),Range]=5
            df.loc[((df[c]>0.5)&(df[c]<=0.6)),Range]=6
            df.loc[((df[c]>0.6)&(df[c]<=0.7)),Range]=7
            df.loc[((df[c]>0.7)&(df[c]<=0.8)),Range]=8
            df.loc[((df[c]>0.8)&(df[c]<=0.9)),Range]=9
            df.loc[((df[c]>0.9)&(df[c]<=1.0)),Range]=10

        for c in columns:
    
            Range=c+'_RANGE'
            df[Range]=0
            df.loc[((df[c]>0)&(df[c]<=5)),Range]=1
            df.loc[((df[c]>5)&(df[c]<=10)),Range]=2
            df.loc[((df[c]>10)&(df[c]<=15)),Range]=3
            df.loc[((df[c]>15)&(df[c]<=20)),Range]=4
            df.loc[((df[c]>20)&(df[c]<=30)),Range]=5
            df.loc[((df[c]>30)&(df[c]<=50)),Range]=6
            df.loc[((df[c]>50)&(df[c]<=100)),Range]=7
            df.loc[((df[c]>100)),Range]=8

        df.drop(['CUST_ID', 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
        'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
        'PURCHASES_FREQUENCY',  'ONEOFF_PURCHASES_FREQUENCY',
        'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
        'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
        'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT' ], axis=1, inplace=True)

        X= np.asarray(df)


        #Padronizando as variáveis
        scale = StandardScaler()
        X = scale.fit_transform(X)

        st.write('## Definindo o número de grupos')  

        #Decidindo o número de clusters
        fig, ax = plt.subplots(figsize=(10, 7)) 
        n_clusters=30
        cost=[]
        for i in range(1,n_clusters):
            kmean= KMeans(i)
            kmean.fit(X)
            cost.append(kmean.inertia_)  

        plt.plot(cost, 'bx-')  
        st.pyplot(fig)

        #Utilizando o Kmeans para decidir os grupos
        kmean= KMeans(6)
        kmean.fit(X)
        labels=kmean.labels_

        clusters=pd.concat([df, pd.DataFrame({'cluster':labels})], axis=1)


        #Analisando os grupos definidos
        
        st.write('## Analisando os 6 grupos definidos')
        for c in clusters:
            grid= sns.FacetGrid(clusters, col='cluster')
            grid.map(plt.hist, c)
            st.pyplot(grid)

        st.write('#### Com os grupos separados pode definir algumas características ')

        st.write('Grupo 0 -> Grupo que faz qualquer tipo de compra com o cartão, parcelado, a vista.')
        st.write('Grupo 1 -> Grupo com pessoas devendo o cartão de crédito.')
        st.write('Grupo 2 -> Grupo que compra a prazo, de forma parcelada, com mais frequência.')
        st.write('Grupo 3 -> Grupo que usa o Cash in Advance, que é uma forma de antecipar o dinheiro.')
        st.write('Grupo 4 -> Grupo que faz as compras com maiores valores.')
        st.write('Grupo 5 -> Grupo que não gasta muito dinheiro')

        st.write('## Visualizando os grupos:')

        dist = 1 - cosine_similarity(X)

        pca = PCA(2)
        pca.fit(dist)
        X_PCA = pca.transform(dist)

        x, y = X_PCA[:, 0], X_PCA[:, 1]

        colors = {0: 'red',
                1: 'blue',
                2: 'green', 
                3: 'yellow', 
                4: 'orange',  
                5:'purple'}

        names = {0: 'Grupo que faz qualquer tipo de compra com o cartão', 
                1: 'Grupo com pessoas devendo o cartão de crédito', 
                2: 'Grupo que compra a prazo, de forma parcelada, com mais frequência', 
                3: 'Grupo que usa o Cash in Advance, que é uma forma de antecipar o dinheiro', 
                4: 'Grupo que faz as compras com maiores valores',
                5: 'Grupo que não gasta muito dinheiro'}
        
        df = pd.DataFrame({'x': x, 'y':y, 'label':labels}) 
        groups = df.groupby('label')

        fig, ax = plt.subplots(figsize=(20, 13)) 

        for name, group in groups:
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=5,
                    color=colors[name],label=names[name], mec='none')
            ax.set_aspect('auto')
            ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
            ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
            
        ax.legend()
        ax.set_title("Customers Segmentation based on their Credit Card usage bhaviour.")
        st.pyplot(fig)

        # Disponibilizando o dataframe em excel


        st.write('Disponibilizando o dataframe clusterizado em Excel')
        df_raw['CLUSTER'] = clusters['cluster']

        df_xlsx = to_excel(df_raw)
        st.download_button(label='📥 Download tabela filtrada em EXCEL',
                            data=df_xlsx ,
                            file_name= 'general_clusterizado.xlsx')
        


if __name__ == '__main__':
	main()