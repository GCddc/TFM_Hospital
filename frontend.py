# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 23:08:23 2021

@author: darze
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from scipy.spatial import distance
import networkx as nx

################################################################################
st.title('¿Qué réspiramos en el Hospital?')
################################################################################
@st.cache
def load_data():
    df=pd.read_excel('Diversidad Hospital_merge_HPy.xlsx', sheet_name='MERGE')
    metadatos=pd.read_excel('Metadatos.xlsx', sheet_name='METADATOS')
    metadatos=metadatos.set_index('Name')
    tax1= df[df.taxlevel==1]
    tax1 = tax1[metadatos.index]
    metadatos['total']=tax1.T
    return df, metadatos

df,metadatos=load_data()

st.header('Tablas de datos')
st.subheader('Secuencias')
st.write(df)
st.subheader('Metadatos')
st.write(metadatos)

#################################################################################
taxlevel = st.sidebar.slider("Tax Level", 2, 6, 6)

tax6= df[df.taxlevel==taxlevel]
del tax6['taxlevel']
del tax6['total']
del tax6['dominio']
del tax6['phylo']
del tax6['clase']
del tax6['orden']
del tax6['familia']
del tax6['genero']
tax6=tax6.set_index('taxon')
tax6=tax6.fillna(0)
st.subheader('Secuencias de nivel %d' % taxlevel)
st.write(tax6)

################################################################################
dfuniques=tax6.nunique()
dftotales=tax6.sum()
frame = { 'Nº secuencias únicas': dfuniques, 'Nº secuencias total': dftotales }

df4 = pd.DataFrame(frame)
df4['unicas'] = 'Nº secuencias únicas'
df4['totales'] = 'Nº secuencias total'
#st.write(df4)

st.header('Gráficas descriptivas')
st.subheader(' Totales vs Unicas  de nivel %d' % taxlevel)
################################################################################
base = alt.Chart(df4.reset_index()).encode(x='index:O')

bar = base.mark_bar(opacity=0.7).encode(y='Nº secuencias total:Q', color='totales:N')

line =  base.mark_bar(opacity=0.7, color='orange').encode(
    y='Nº secuencias únicas:Q',
    color='unicas:N'

)

st.write((bar + line).properties(width=600).resolve_scale(y = 'independent'))

################################################################################

meta=metadatos


f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(7, 12), sharex=True)
sns.barplot(x=meta.index, y=meta['Chao-1'],color="red", ax=ax1)
ax1.axhline(0, color="k", clip_on=False)
ax1.set_ylabel("Chao-1")
ax1.set_xlabel('')
sns.barplot(x=meta.index, y=meta['Shannon H Index'], color="green", ax=ax2)
ax2.axhline(0, color="k", clip_on=False)
ax2.set_ylabel('Shannon')
ax2.set_xlabel('')
sns.barplot(x=meta.index, y=meta['Simpson Index'], color="yellow", ax=ax3)
plt.xticks(rotation=60)
ax3.axhline(0, color="k", clip_on=False)
ax3.set_ylabel('Simpson')
ax3.set_xlabel('')
sns.barplot(x=meta.index, y=meta['total'],color="magenta", ax=ax4)
ax4.axhline(0, color="k", clip_on=False)
ax4.set_ylabel("Total")
ax4.set_xlabel('')
sns.despine(bottom=True)
#plt.setp(f.axes, yticks=[])
#plt.tight_layout(h_pad=2)
f.suptitle('Indices de diversidad')


st.pyplot(f)

################################################################################

st.subheader('Proporción de los Phylos más relevantes')

tax2= df[df.taxlevel==2]
del tax2['taxlevel']
del tax2['total']
del tax2['dominio']
del tax2['phylo']
del tax2['clase']
del tax2['orden']
del tax2['familia']
del tax2['genero']
tax2=tax2.set_index('taxon')
tax2_norm=tax2/tax2.sum()*100
tax2_norm_orden = tax2_norm.sort_values(by=['OR-S'],ascending=False)

f, ax = plt.subplots()
tax2_norm_orden[0:13].T.plot.bar(ax=ax, stacked=True)
plt.legend(bbox_to_anchor=(1,1),ncol=1)
st.pyplot(f)


################################################################################

st.header('PCA')

normalizadores = ['Ninguno', 'Log(x+1)', 'StandardScaler']

normalizador = st.selectbox("Normalizador", normalizadores, index=1)

pca = PCA()
if normalizador == 'Ninguno':
    projected = pca.fit_transform(tax6)
elif normalizador == 'Log(x+1)':
    normalized = FunctionTransformer(np.log1p).fit_transform(tax6)
    projected = pca.fit_transform(normalized)
elif normalizador == 'StandardScaler':
    normalized = StandardScaler().fit_transform(tax6)
    projected = pca.fit_transform(normalized)

################################################################################
st.subheader('Varianza explicada')
f, ax = plt.subplots()
ax.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
st.pyplot(f)

st.write("Varianza explicada por las 3 primeras componentes: %.03f" % pca.explained_variance_ratio_[:3].sum())

################################################################################

first_four = pd.DataFrame(data =projected[:, :4], index=tax6.index, columns = ['PC%d' % n for n in range(1, 5) ])
first_four['phylo'] =  df[df.taxlevel == taxlevel].set_index('taxon').phylo #recupero la columna phylo desde el df no el tax6

st.write(first_four)

################################################################################

st.subheader('Pairplot de los cuatro primeros componentes')

st.pyplot(sns.pairplot(first_four, kind='scatter'))
st.subheader('Phylos en PC1 y PC2')
cm = plt.get_cmap('gist_rainbow')
num_colors = len(first_four.phylo.unique())
fig, ax =plt.subplots()
ax.set_prop_cycle('color', [cm(1.*i/num_colors) for i in range(num_colors)])
for phylo in first_four.phylo.unique():
    ix=np.where(first_four.phylo==phylo)
    ax.scatter(first_four.PC1.values[ix],first_four.PC2.values[ix],label=phylo,alpha=0.5)
 
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.legend(bbox_to_anchor=(1,1), ncol=2)
st.pyplot(fig)


PC_names = [f'PC{n+1}' for n in range(len(pca.components_))]
Componentes=pd.DataFrame(
    data    = pca.components_,
    columns = tax6.columns,
    index   = PC_names
)

first_4 = Componentes[0:4]
first_4_meta = first_4.T.join(metadatos).reset_index()
################################################################################
st.subheader('Pairplot de los loadings de los cuatro primeros componentes')

st.pyplot(sns.pairplot(first_4.T, kind='scatter'))

color=st.selectbox('color',metadatos.select_dtypes(object).columns, 0)
shape=st.selectbox('shape',metadatos.select_dtypes(object).columns, 1)


st.subheader('Loadings de los dos primeros componentes')
scatter = alt.Chart(first_4_meta).mark_point().encode(
    x='PC1',
    y='PC2',
    color=color,
    shape=shape
).properties(
    width=600,
    height=400
)

text = scatter.mark_text(
    align='left',
    baseline='middle',
    dx=7
).encode(
    text='index'
)

st.write(scatter + text)

################################################################################



st.header('Network')

umbral_bc = st.slider("Umbral distancia Bray-Curtis", 0., 1., value=0.55, step=0.05) #0.55
umbral_secuencias = st.slider("Porcentaje minimo de abundancia de secuencia",
                              0, 100, 3)/100 #0.03

BC=pd.DataFrame([[distance.braycurtis(tax6[i], tax6[j]) for i in tax6 ] for j in tax6], 
               columns=tax6.columns,index=tax6.columns)

tax6norm = tax6 / tax6.sum()
tax6b = tax6norm.where(tax6norm > umbral_secuencias, 0)  # Seleccionar los que superan un umbral
tax6b = tax6b[tax6b.T.any()]  # Eliminar las filas vacías
tax6b = tax6b * 4 # Incrementar valores absolutos para que sean comparables con los de los lugares de muestreo

tax6b[tax6b.index] = 0  # Añadir columnas con los nombres de los microorganismos
adj = pd.concat([tax6b,BC<umbral_bc]).fillna(0) # Añadir filas con la adyacencia de Bray-Curtis

def colorize(node):
    if node in tax6.columns:
        return '#ebeb34'
    return '#34b7eb'

G = nx.from_pandas_adjacency(adj)
G.name = "Hospitales"

f, ax = plt.subplots(figsize=(20, 20))
pos=nx.spring_layout(G, weight=None)
node_color = [colorize(x) for x in G.nodes()]


weights = np.array(list(nx.get_edge_attributes(G, 'weight').values()))

nx.draw_networkx_nodes(G, pos=pos, node_size=1500,node_color=node_color, alpha=0.5)
nx.draw_networkx_edges(G, pos=pos, width=weights, edge_color=weights, edge_cmap=plt.get_cmap('viridis'))
nx.draw_networkx_labels(G, pos=pos)

st.pyplot(f)
st.text(nx.info(G))

################################################################################

st.header('Krona')

with open('krona/krona_hospital.krona.html') as html:
    components.html(html.read(), height=600)