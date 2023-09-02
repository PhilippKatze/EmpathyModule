import pandas as pd
import numpy as np

import os
import sys
import wave
import struct
import sounddevice as sd


import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
from pvrecorder import PvRecorder

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from IPython.display import Audio

import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


path = ""
data, sample_rate = "", 0.1

Ravdess = "data/"
Crema = "/kaggle/input/cremad/AudioWAV/"
Tess = "/kaggle/input/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data/"
Savee = "/kaggle/input/surrey-audiovisual-expressed-emotion-savee/ALL/"

def trainModel():
    ravdess_directory_list = os.listdir(Ravdess)

    file_emotion = []
    file_path = []
    for dir in ravdess_directory_list:
        # as their are 20 different actors in our previous directory we need to extract files for each actor.
        actor = os.listdir(Ravdess + dir)
        for file in actor:
            part = file.split('.')[0]
            part = part.split('-')
            # third part in each file represents the emotion associated to that file.
            file_emotion.append(int(part[2]))
            file_path.append(Ravdess + dir + '/' + file)

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

    # changing integers to actual emotions.
    Ravdess_df.Emotions.replace(
        {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'},
        inplace=True)
    Ravdess_df.head()

    path = np.array(Ravdess_df.Path)[1]
    data, sample_rate = librosa.load(path)
    print(Ravdess_df)

    X, Y = [], []
    for path, emotion in zip(Ravdess_df.Path, Ravdess_df.Emotions):
        feature = get_features(path)
        for ele in feature:
            X.append(ele)
            # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
            Y.append(emotion)

    Features = pd.DataFrame(X)
    Features['labels'] = Y
    Features.to_csv('features.csv', index=False)
    Features.head()

    X = Features.iloc[:, :-1].values
    Y = Features['labels'].values

    encoder = OneHotEncoder()
    Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()

    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = Sequential()
    model.add(
        Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(x_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))

    model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))

    model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(units=8, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()


    rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)
    history = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), callbacks=[rlrp])

    print("Accuracy of our model on test data : ", model.evaluate(x_test, y_test)[1] * 100, "%")

    pred_test = model.predict(x_test)
    y_pred = encoder.inverse_transform(pred_test)

    y_test = encoder.inverse_transform(y_test)

    df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
    df['Predicted Labels'] = y_pred.flatten()
    df['Actual Labels'] = y_test.flatten()

    df.head(10)

    model.save("model.h5")
    np.save('X_train.npy', x_train)


def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(y=data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps= pitch_factor)


def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    return result


def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res1 = extract_features(data)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2))  # stacking vertically

    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3))  # stacking vertically

    return result

def predictEmotion(path):

    print("predicting")
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(np.array(['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']).reshape(1, -1))

    x_train = np.load('X_train.npy')
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    ModelFilePath = 'model'
    Voice_emotion_Detection = keras.models.load_model(ModelFilePath)

    data_, sample_rate_ = librosa.load(path)

    X_ = np.array(extract_features(data_))
    X_ = scaler.transform(X_.reshape(1, -1))
    pred_test_ = Voice_emotion_Detection.predict(np.expand_dims(X_, axis=2))
    print(pred_test_)
    y_pred_ = encoder.inverse_transform(pred_test_)
    print(y_pred_[0][0])  # emotion prediction

    for value, emotion in zip(pred_test_[0], encoder.categories_[0]):
        print(emotion, f"{value:.10f}")  # predicting values for each emotion




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('press any key to stop recording')

    #freq = 44100
    #duration = 5
    #recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
    #sd.wait()

    #inputfile = librosa.load('input')
    print('audio recorded')
    predictEmotion('data/Actor_23/03-01-02-01-02-01-23.wav')
    #trainModel()


