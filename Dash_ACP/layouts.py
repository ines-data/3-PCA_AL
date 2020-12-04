import dash_core_components as dcc
import dash_html_components as html
from app import app
import pandas as pd 
import numpy as np
import plotly.graph_objs as go
import dash_table
from dash.dependencies import Output, Input
import plotly.graph_objects as go  
# import figure factory
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

timeData = pd.read_csv("timesData.csv")
timeData.country = timeData['country'].replace(['Unisted States of America', 'Unted Kingdom', 'Austria'],
                                   ['United States of America','United Kingdom', 'Australia'])
df = timeData
print(df.isnull().values.any())
df = df.dropna()
df = timeData[timeData['year']==2016]
df = df[:50]
df['world_rank'] = df['world_rank'].replace(['=39'],'39')
df['world_rank'] = df['world_rank'].replace(['=44'],'44')
df['world_rank'] = df['world_rank'].replace(['=47'],'47')
df.world_rank = pd.to_numeric(df.world_rank)
df.international = pd.to_numeric(df.international)
df.income = [float(each.replace('-', 'NaN')) for each in df.income]
df= df.dropna()
df.total_score = pd.to_numeric(df.total_score)
df['num_students'] = [float(each.replace(',','.')) for each in df.num_students]
df.international_students = [int(each.replace('%','')) for each in df.international_students]
def convertRatio (x):
    a, b= x.split(':')
    c = int(a)/int(b)
    return round(c,2)

df.female_male_ratio = df['female_male_ratio'].apply(convertRatio)

df2 = df[['world_rank','teaching','international','research','citations','income', 'total_score','num_students']]
# prepare data
df2['index']= np.arange(1, len(df2)+1) # ajout d'une colonne index pour avoir la nuance des couleurs à droite du graphe

dff = df.corr(method="spearman")

########  ACP / PCA  #########
x = df[['teaching','international','research','citations','income','total_score','num_students',
       'student_staff_ratio','international_students','female_male_ratio']]
y = df['world_rank']

X4= StandardScaler().fit_transform(x)
pca4 = PCA(n_components=2)
pca4.fit(X4)
PCA4 = pca4.fit_transform(X4)


def get_menu():
    menu = html.Div(style={'font-size': '20px', 'background-color':'#EBF0F5', 'textAlign': 'center', 'color':'rgba(0,0,7,0.7)'}, children=[ 

        dcc.Link('Analyse des données|', href='/Analysedesdonnées', className='mb-3'),

        dcc.Link('Analyse en Composantes Principales', href='/AnalyseenComposantesPrincipales', className="mb-3")
    ], className="rows")
    return menu

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

def generate_table2(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

def generate_table3(dataframe):
    return dash_table.DataTable(
    data=df.to_dict('records'),
    columns=[{'id': c, 'name': c} for c in df.columns],
    page_size=10
    )

fig1 = ff.create_scatterplotmatrix(df2,        diag='histogram',
                                  colormap='Viridis',
                                  colormap_type='cat',
                                  height=800, width=1400)
fig = ff.create_scatterplotmatrix(df2, diag='box',index = 'index',colormap='Portland',colormap_type='cat',height=700, width=700)
fig2 = px.scatter(df2, x="world_rank", y="research", color="world_rank")
fig3 = px.scatter(df2, x="world_rank", y="teaching", color="world_rank")
fig4 = px.scatter(df2, x="world_rank", y="citations", color="world_rank")
fig5 = px.scatter(df2, x="research", y="teaching", color="world_rank")

text = '''
On va analyser les données des 50 premières Universités et essayer de voir s'il y a 
des correlations entre certains critères :
- Dans un premier temps on va analyser les données de manières classique.
- Dans un deuxième temps on utiler la méthode de l'analyse des composantes principals.
''' 
text1 = '''
On remaque une correlation entre :
- Le classement mondial des unvinersités et le score universitaire pour la recherche (volume, revenu et réputation) (research)
- Le classement mondial des universités et le score universitaire pour l'enseignement (l'environnement d'apprentissage) (teaching)
- Le classement mondial des universités et score universitaire pour les citations (influence de la recherche) (citations)
- le score universitaire pour l'enseignement (l'environnement d'apprentissage) (teaching) et le score universitaire pour la recherche (reaserch)
'''
text2 = '''
L’ analyse en composantes principales (ACP) , ou principal component analysis (PCA) en anglais, permet d’analyser et de visualiser un jeu de données contenant des individus décrits par plusieurs variables quantitatives.

C’est une méthode statistique qui permet d’explorer des données dites multivariées (données avec plusieurs variables). Chaque variable pourrait être considérée comme une dimension différente. Si vous avez plus de 3 variables dans votre jeu de données, il pourrait être très difficile de visualiser les données dans une “hyper-espace” multidimensionnelle.

L’analyse en composantes principales est utilisée pour extraire et de visualiser les informations importantes contenues dans une table de données multivariées. L’ACP synthétise cette information en seulement quelques nouvelles variables appelées composantes principales. Ces nouvelles variables correspondent à une combinaison linéaire des variables originels. Le nombre de composantes principales est inférieur ou égal au nombre de variables d’origine.

L’information contenue dans un jeu de données correspond à la variance ou l’inertie totale qu’il contient. L’objectif de l’ACP est d’identifier les directions (i.e., axes principaux ou composantes principales) le long desquelles la variation des données est maximale.

En d’autres termes, l’ACP réduit les dimensions d’une donnée multivariée à deux ou trois composantes principales, qui peuvent être visualisées graphiquement, en perdant le moins possible d’information.
'''

layout1 = html.Div([
    html.H1('Classement universitaire', style={'textAlign': 'center', 'color':'rgba(0,0,7,0.7)', 'border':'3px double black'}),
    html.Div(dcc.Markdown(text, style={'font-size':'18px'})),
    get_menu(),
    html.H2(children='Tableau des 50 premières universités au classement mondial',style={'padding': '30px'}),
    generate_table3(df),
    dcc.Graph(
        id = 'id3',
        figure=fig1
    ),
    html.H2(children='Tableau des correlations',style={'padding': '30px'}),
    generate_table3(dff),
    html.Div(dcc.Markdown(text1, style={'font-size':'18px'})),
    html.Div(html.H3('Correlation entre le rang universitaire et le score universitaire pour la recherche'),style={'padding': '30px'}),
    dcc.Graph(
        id = 'id3',
        figure=fig2
    ),
    html.Div(html.H3('Correlation entre le rang universitaire et le score universitaire pour l\'enseignement'),style={'padding': '30px'}),
    dcc.Graph(
        id = 'id3',
        figure=fig3
    ),
    html.Div(html.H3('Correlation entre le rang universitaire et le score universitaire pour les citations'),style={'padding': '30px'}),
    dcc.Graph(
        id = 'id3',
        figure=fig4
    ),
    html.Div(html.H3('Correlation entre le score universitaire pour la recherche et le score universitaire pour l\'enseignement'),style={'padding': '30px'}),
    dcc.Graph(
        id = 'id3',
        figure=fig5
    ),
    #lineCharts(df1),
    #bar_Charts(df1),
    #html.Div(html.H2(children=text1)),
    html.Div(id='id1'),
    dcc.Link('Go to Analyse en Composantes Principales', href='/AnalyseenComposantesPrincipales')
])

layout2 = html.Div([
    html.H1('Classement universitaire', style={'textAlign': 'center', 'color':'rgba(0,0,7,0.7)', 'border':'3px double black'}),
    html.H3('Analyse en Composantes Principales (ACP)'),
    html.Div(dcc.Markdown(text2, style={'font-size':'18px'})),
    get_menu(),
    html.H2(children='Tableau des 50 premières universités au classement mondial',style={'padding': '30px'}),
    generate_table3(df),
    html.Div([
        html.Img(src=app.get_asset_url('ACP.png'), height=700, width=700, style={'padding-left':'25vw'})
        #html.Div(style = {'padding-left': '30vw', 'height':'180', 'width':'150'},children = [html.Img(id='image',src=app.get_asset_url('ACP.png'))]
    ]),#, style ={'height':'180', 'width':'150'})

    #]),
    #dcc.Graph(
    #    id = 'id3',
    #    figure=fig3
    #),
    #dcc.Graph(
    #    id = 'id4',
    #    figure=fig2
    #),
    #dcc.Graph(
    #    id='id5',
    #   figure=fig
    #),
    html.Div(id='id3'),
    dcc.Link('Go to Analyse des données', href='/Analysedesdonnées')
])