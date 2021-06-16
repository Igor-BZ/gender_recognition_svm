
#https://www.kaggle.com/rajmehra03/a-complete-tutorial-onpredictive-modeling-acc-99
'''IMPORTACION DE MODULOS'''
# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns #data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
import missingno as msno

#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
#%matplotlib inline  
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#import the necessary modelling algos.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV

#preprocess.
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer
#imputer = SimpleImputer(missing_values=np.nan, strategy='mean')


#Cargar  DataFrame
train=pd.read_csv(r'training_data.csv')
df=train.copy()
df.isnull().any() #comprobar que no hay datos faltantes

#########################################################################################################################################################

'''
ANALISIS DE DATOS EXPLORATORIOS
Análisis univariante
todas las características son "numéricas", la forma más razonable de representarlas sería un "histograma" o un "diagrama de caja".
útil para la detección de valores atípicos. Por lo tanto, además de trazar un diagrama de caja y un histograma para cada columna o característica
Se realiza entonces analisis estadistico
Tenga en cuenta que tenemos el mismo número de observaciones para los 'hombres' y las 'mujeres'. Por tanto, es un problema de clases equilibrado.
'''
df.describe()
#feature=caracteristica
def calcular_limites(feature): 
    q1,q3=df[feature].quantile([0.25,0.75])  #q1->primercuartil->Q25; q3->tercercuartil->Q75
    iqr=q3-q1#rango intercuartil=iferencia entre el tercer y el primer cuartil, Mediante esta medida se eliminan los valores extremadamente alejados.
    #En una distribución, encontramos la mitad de los datos, el 50 %, ubicados dentro del rango intercuartílico. 
    rang=1.5*iqr
    #Q25-RANG= Limite Inferior; Q75+RANG=Limite Superior
    return(q1-rang,q3+rang)

def grafico(feature):
    fig,axes=plt.subplots(1,2) #crea una figura (nfilas, ncolumnas) osea dos graficos en una fila
    sns.boxplot(data=df,x=feature,ax=axes[0],color='#2F75B5') #AX=se coloca en el grafico 1; grafico de 'jeringas'
    sns.distplot(a=df[feature],ax=axes[1],color='#2F75B5')#A=serie, datos observados; histograma y linea de distribucion
    fig.set_size_inches(15,5) #tamaño
    
    lower,upper = calcular_limites(feature)
    l=[df[feature] for i in df[feature] if i>lower and i<upper] #cantidad de valores atipicos
    print("Número de puntos de datos restantes si se eliminan los valores atípicos : ",len(l))
    fig.savefig(feature.upper()+'grafico') #guardar graficos 
    plt.clf()

'''
Conclusiones del Analisis de Grafico MEANFREC

1) En primer lugar, tenga en cuenta que los valores cumplen con los observados en el marco de datos del método de descripción.
2) Tenga en cuenta que tenemos un par de valores atípicos w.r.t. a la regla del cuartil 1.5 (representado por un 'punto' en el diagrama de caja). Quitar estos puntos de datos o valores atípicos nos deja con alrededor de 3104 valores.
3) Observe también en la gráfica de distribución que la distribución parece estar un poco sesgada, por lo que podemos normalizarla para hacer que la distribución sea un poco más simétrica.
4) ÚLTIMO TENGA EN CUENTA QUE UNA DISTRIBUCIÓN DE COLA IZQUIERDA TIENE MÁS SALIDAS EN EL LADO DEBAJO DE Q1 COMO SE ESPERA Y UNA COLA DERECHA POR ENCIMA DE Q3.

HAY QUE HACER ESTE ANALISIS EN TODOS #2 LOS GRAFICOS IDEALMENTE
'''

#########################################################################################################################################################

'''
Análisis bivariado

1) Correlacion entre caracteristicas
Se analizó la correlación entre diferentes características. Para hacerlo, se ha trazado un 'mapa de calor' que visualiza claramente la correlación entre diferentes características.
'''
def MatrizCorrelacion():
    temp=[]
    for i in df.label: 
        if i =='male':  
                temp.append(1)
        else:   
                temp.append(0)
    df['label']=temp # cambia hombres por un 1 y mujeres por 0, funcion correlacion no lee str

    #matriz de correlacion
    cor_mat=df[:].corr()
    mask=np.array(cor_mat) #se define array con los datos del df
    mask[np.tril_indices_from(mask)] = False #Devuelve los índices del triángulo inferior de una matriz (n, m).
    #dibujar
    fig2=plt.gcf()#Get the current figure, If no current figure is available then one is created with the help of the figure() function.
    fig2.set_size_inches(30,12)
    sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True,cmap="YlGnBu")#crea mapa de calor
    fig2.savefig('Matriz de Correlacion') #guardar graficos 

'''
Conclusiones del Analisis del Mapa de Calor


1) La meanfreq está moderadamente relacionada con el Genero.
        ALGO IMPORTANTE
2) IQR y el Genero tienden a tener una fuerte correlación positiva.
        IMPORTANTE
3) La entropía espectral también está muy correlacionada con el Genero, mientras que sfm está moderadamente relacionada con este.
        IMPORTANTE
4) la asimetría y la curtosis no están muy relacionadas con el genero. 
        NO IMPORTANTE
5) meanfun tiene una correlación muy negativa con el genero.
        IMPORTANTE PERO AL REVES
6) El centroide y la mediana tienen una alta correlación positiva como se esperaba de sus fórmulas.
        --
7) TENGA EN CUENTA QUE MEANFREQ Y CENTROID SON EXACTAMENTE LAS MISMAS CARACTERÍSTICAS QUE LAS FÓRMULAS Y LOS VALORES TAMBIÉN. POR LO QUE SU CORELACIÓN ES PERFCET 1. EN ESE CASO PODEMOS CAER CUALQUIER COLUMNA. tenga en cuenta que el centroide en general tiene un alto grado de correlación con la mayoría de las otras características.
ASÍ QUE BAJARÉ LA COLUMNA 'CENTROIDE'.
        --
8) sd está altamente relacionada positivamente con sfm y también lo está sp.ent con sd.
        IMPORTANTE, sfm/sp.ent->sd->iqr->genero
9) kurt y skew también están muy correlacionados.
        -- 
10) meanfreq está muy relacionado con medaina s así como Q25.
        IMPORTANTE, esta meanfrec
11) IQR está altamente correlacionado con sd.
        IMPORTANTE, por iqr
12) Finalmente, la relación de uno mismo, es decir, de una característica consigo misma, es igual a 1 como se esperaba.

Por lo tanto, es importante tener en cuenta:
    -meanfreq
    -IQR
    -sp.ent
    -(-meanfun)
    -sd
    -sfm
    -q25
    -medaina s

Tenga en cuenta que se pueden descartar algunas características altamente relacionadas, ya que agregan redundancia al modelo
En el caso de características altamente correlacionadas, podemos usar técnicas de reducción de dimensionalidad como Análisis de Componentes Principales (PCA) para reducir nuestro espacio de características.
(el profe hablo de esto en clases :o)
'''

df.drop('centroid',axis=1,inplace=True)
df.drop('Q75',axis=1,inplace=True)
df.drop('skew',axis=1,inplace=True)
df.drop('kurt',axis=1,inplace=True)
df.drop('mode',axis=1,inplace=True)
df.drop('minfun',axis=1,inplace=True)
df.drop('maxfun',axis=1,inplace=True)
df.drop('meandom',axis=1,inplace=True)
df.drop('mindom',axis=1,inplace=True)
df.drop('maxdom',axis=1,inplace=True)
df.drop('dfrange',axis=1,inplace=True)
df.drop('modindx',axis=1,inplace=True)

#########################################################################################################################################################

'''
Caracteristicas segun Genero
ejecutar para cada caracteristica
'''

def caract_gen(feature):
    sns.factorplot(data=df,y=feature,x='label',kind='strip',palette='YlGnBu')
    fig=plt.gcf()
    fig.set_size_inches(7,7)
    fig.savefig(feature.upper()+'caract-gen') #guardar graficos 

'''
Nuevamente, una gran diferencia en mujeres y hombres significa frecuencia fundamental. 
Esto es evidente en el mapa de calor que muestra claramente la alta correlación entre meanfun y la 'etiqueta'.
Ahora pasamos al análisis de diferentes características por parejas. Dado que todas las características son continuas,
la forma más razonable de hacerlo es trazar los gráficos de dispersión para cada par de características.
También he distinguido machos y hembras en la misma parcela, lo que hace que sea un poco más fácil comparar la variación de características dentro de las dos clases. 
'''
def comp_caract():
        g = sns.PairGrid(df[['meanfreq','median','sd','Q25','IQR','sp.ent','sfm','meanfun','label']], hue = "label",palette='YlGnBu')
        g = g.map(plt.scatter).add_legend()
        g.savefig('ComparacionCaracts') #guardar graficos 

#########################################################################################################################################################
'''
Tratamiento de valores atípicos

En esta sección me he ocupado de los valores atípicos. Tenga en cuenta que descubrimos los valores atípicos potenciales en la sección de "análisis univariante".

Ahora, para eliminar esos valores atípicos, podemos eliminar los puntos de datos correspondientes o imputarlos con alguna otra cantidad estadística como la mediana (robusta a los valores atípicos), etc.

Por ahora, eliminaré todas las observaciones o puntos de datos que sean atípicos para "cualquier" característica. Tenga en cuenta que esto reduce sustancialmente el tamaño del conjunto de datos.
'''
#BUSCAR OTRO METODO DE TRATAR OUTLIERSS
def eliminar_outlier():
        for col in df.columns:
                try:
                        lower,upper=calcular_limites(col)
                        df = df[(df[col] >lower) & (df[col]<upper)]     
                except:
                        pass

#########################################################################################################################################################
'''
Estandarice las características eliminando la media y escalando a la varianza de la unidad
StandardScaler():
La puntuación estándar de una muestra xse calcula como:

    z = (x - u) / s

    SE APLICA DESVIACION ESTANDAR PARA NORMALIZAR LOS DATOS UWU
fIT_TRANSFORM=
Fit to data, then transform it.

Fits transformer to X and y with optional parameters fit_params and returns a transformed version of X.

Xarray-like of shape (n_samples, n_features)
yarray-like of shape (n_samples,) or (n_samples, n_outputs),
X_newndarray array of shape (n_samples, n_features_new)
'''
def Normalizar():
        temp_df=df
        scaler=StandardScaler()
        scaled_df=scaler.fit_transform(temp_df.drop('label',axis=1))
        X=scaled_df
        Y=df['label'].to_numpy()#Convert the frame to its Numpy-array representation.
        x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=42) #Split arrays or matrices into random train and test subsets
#test_size=proporción del conjunto de datos para incluir en la división de prueba. 
#random_state=Controla la mezcla aplicada a los datos antes de aplicar la división. Pase un int para una salida reproducible a través de múltiples llamadas a funciones 



 

def Modelado(modelo):
        clf_svm=SVC() #C-Support Vector Classification.
        clf_svm.fit(x_train,y_train)#fit(X, y[, sample_weight])Fit the SVM model according to the given training data.
        pred=clf_svm.predict(x_test)#Perform classification on samples in X.
        print(accuracy_score(pred,y_test))#his function computes subset accuracy


Modelado(SVC(kernel='linear',C=10))
Modelado(SVC(kernel='rbf',C=10))
Modelado(SVC(kernel='poly',C=10))
Modelado(SVC(kernel='sigmoid',C=10))

def MejorSVM():
        params_dict={'C':[0.001,0.01,0.1,1,10,100],'gamma':[0.001,0.01,0.1,1,10,100],'kernel':['linear','rbf','poly','sigmoid']}
        clf=GridSearchCV(estimator=SVC(),param_grid=params_dict,scoring='accuracy',cv=10)
        clf.fit(x_train,y_train)

        acierto=clf.best_score_
        parametros=clf.best_params_
        return aciertos,parametros
