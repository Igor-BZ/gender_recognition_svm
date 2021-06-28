
'''IMPORTACION DE MODULOS'''
# Ignore  the warnings
from typing import ValuesView
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns 
import missingno as msno

#configure 
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)
import os

#import the necessary modelling algos.
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import NuSVC

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV

#preprocess.
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer

#audio
import pyaudio
import wave

#Cargar  DataFrame
train=pd.read_csv(r'training_data.csv')
df=train.copy()

#Elimina Columnas Innecesarias o Redundantes
def Eliminar_Columnas():
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
    df.drop('sfm',axis=1,inplace=True)
    df.drop('sp.ent',axis=1,inplace=True)
    df.drop('meanfreq',axis=1,inplace=True)
    df.drop('median',axis=1,inplace=True)

#Caracteristicas segun Genero
def Car_Genero(feature):
    sns.factorplot(data=df,y=feature,x='label',kind='strip',palette='YlGnBu')
    plt.title(feature.upper())
    fig=plt.gcf()
    fig.set_size_inches(7,7)
    fig.savefig(feature.upper()+'caract-gen') #guardar graficos 

             

def Normalizar():
        temp_df=df
        scaler=StandardScaler()
        scaled_df=scaler.fit_transform(temp_df.drop('label',axis=1))
        X=scaled_df
        Y=df['label'].to_numpy()
        x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=42) 
        #test_size=proporción del conjunto de datos para incluir en la división de prueba. 
        #random_state=Controla la mezcla aplicada a los datos antes de aplicar la división. Pase un int para una salida reproducible a través de múltiples llamadas a funciones 
        return x_train,x_test,y_train,y_test

def Modelado(modelo):
        Normalizar()
        clf_svm=modelo 
        clf_svm.fit(x_train,y_train)
        pred=clf_svm.predict(x_test)
        print(accuracy_score(pred,y_test))
        return clf_svm

def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "test_audio.wav"

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("* recording")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def Obtener_Valores_Voz():
        record_audio()
        os.system('"Praat.exe" --run extract_freq_info.praat')
        file = open('output.txt','r') 
        values = file.readline()
        values = values.split(', ')
        values=values[0:4]
        for x in range(0,4):
                values[x] = float(values[x])/1000
        values=np.array(values)
        valores_voz=[values]
        return valores_voz

#************   MAIN   ******************
Eliminar_Columnas()

x_train,x_test,y_train,y_test=Normalizar()
clf=Modelado(SVC(kernel='rbf'))

voz=Obtener_Valores_Voz()
clf.predict(voz)


