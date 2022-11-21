
import streamlit as st
import numpy as np
import pandas as pd
from pydub import AudioSegment
import librosa
import pickle
from sklearn.preprocessing import StandardScaler

mood = "Waiting for prediction"

def extract():
    features = []
    audio, sample_rate = librosa.load( '/content/audio.mp3', res_type = 'kaiser_fast')
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mean_mfcc = np.mean(mfcc,axis=1)
    #ms = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=40)
    #mean_ms = np.mean(ms.T,axis=0)
    features.append(mean_mfcc)
    features_data = pd.DataFrame(features)
    return features_data

st.title("Audio Mood Analysis")

file = st.file_uploader("audio_file", type="mp3", label_visibility="visible")
st.write(mood)

if file is not None:
  file_var = AudioSegment.from_mp3(file) 
  file_var.export('audio.mp3', format='mp3')
  test_features = extract()
  test_features.drop(columns = test_features.columns[0], inplace = True)
  #print(test_features.head())
  Test_X = test_features.iloc[: ,:].values
  print(Test_X.shape)
  scaler = StandardScaler()
  Test_X = scaler.fit_transform(Test_X)
  Test_X = np.expand_dims(Test_X, axis=2)
  print(Test_X.shape)
  tmodel = pickle.load(open("tmp.pkl","rb"))
  p = tmodel.predict(Test_X)
  l = p.tolist()[0]
  qc1=l.index(max(l))
  mood = "Q"+str(qc1+1)
  st.write("Prediction: ",mood)
