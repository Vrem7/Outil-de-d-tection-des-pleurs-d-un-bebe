import os
import numpy as np
import librosa
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def caracteristiques_audio(fichier_audio, taille_segment):
    sr = 44100
    donnees_audio, _ = librosa.load(fichier_audio, sr=sr)
    spectrogramme = librosa.feature.melspectrogram(y=donnees_audio, sr=sr)
    mfccs = librosa.feature.mfcc(y=donnees_audio, sr=sr, n_mfcc=13)

    segments_spectrogramme = []
    debut = 0
    while debut + taille_segment <= spectrogramme.shape[1]:
        segment = spectrogramme[:, debut:debut+taille_segment]
        segments_spectrogramme.append(segment.flatten())
        debut += taille_segment

    segments_mfccs = []
    debut = 0
    while debut + taille_segment <= mfccs.shape[1]:
        segment = mfccs[:, debut:debut+taille_segment]
        segments_mfccs.append(segment.flatten())
        debut += taille_segment

    return segments_spectrogramme, segments_mfccs

def filtrer_segments_sonores(segments, seuil_silence=20):
    segments_sonores = []
    for segment in segments:
        amplitude = np.max(segment)
        if amplitude > seuil_silence:
            segments_sonores.append(segment)
    return segments_sonores

dossier_pleure = 'C:\\Users\\AskyM\\OneDrive\\Documents\\Projet_bebe\\Son\\pleure'
dossier_non_pleure = 'C:\\Users\\AskyM\\OneDrive\\Documents\\Projet_bebe\\Son\\nonpleure'
dossier_sauvegarde = 'C:\\Users\\AskyM\\OneDrive\\Documents\\Projet_bebe\\Save'

fichiers_pleure = [os.path.join(dossier_pleure, nom_fichier) for nom_fichier in os.listdir(dossier_pleure)]
fichiers_non_pleure = [os.path.join(dossier_non_pleure, nom_fichier) for nom_fichier in os.listdir(dossier_non_pleure)]

taille_segment = 80
donnees_segments_combined = []
etiquettes_segments = []

for fichier in fichiers_pleure:
    segments_spectrogramme, segments_mfccs = caracteristiques_audio(fichier, taille_segment)
    segments_combined = np.concatenate((segments_spectrogramme, segments_mfccs), axis=1)
    donnees_segments_combined.extend(segments_combined)
    etiquettes_segments.extend([1] * len(segments_spectrogramme))

for fichier in fichiers_non_pleure:
    segments_spectrogramme, segments_mfccs = caracteristiques_audio(fichier, taille_segment)
    segments_combined = np.concatenate((segments_spectrogramme, segments_mfccs), axis=1)
    donnees_segments_combined.extend(segments_combined)
    etiquettes_segments.extend([0] * len(segments_spectrogramme))

donnees_segments_combined = np.array(donnees_segments_combined)
etiquettes_segments = np.array(etiquettes_segments)

X_train, X_test, y_train, y_test = train_test_split(donnees_segments_combined, etiquettes_segments, test_size=0.2, random_state=42)

modele = Sequential()
modele.add(Dense(units=128, activation='relu', input_shape=(11280,)))
modele.add(Dropout(0.3))
modele.add(Dense(units=64, activation='relu'))
modele.add(Dropout(0.3))
modele.add(Dense(units=32, activation='relu'))
modele.add(Dropout(0.3))
modele.add(Dense(units=1, activation='sigmoid'))

modele.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

modele.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))

perte, precision = modele.evaluate(X_test, y_test)
print(f"Perte de test : {perte:.4f}, Précision de test : {precision:.4f}")

nom_modele = 'Train.h5'
Save = os.path.join(dossier_sauvegarde, nom_modele)
modele.save(Save)


while True:
    nom_fichier = input("Entrez le nom d'un fichier audio dans le dossier de test : ")
    fichier_test = os.path.join('C:\\Users\\AskyM\\OneDrive\\Documents\\Projet_bebe\\Son\\test', nom_fichier)
    if os.path.exists(fichier_test):
        segments_spectrogramme_test, segments_mfccs_test = caracteristiques_audio(fichier_test, taille_segment)
        segments_test = np.concatenate((np.array(segments_spectrogramme_test), np.array(segments_mfccs_test)), axis=1)
        predictions = modele.predict(np.array(segments_test))
        
        nombre_predictions_positives = np.sum(predictions)
        nombre_total_predictions = len(predictions)
        
        pourcentage_predictions_positives = (nombre_predictions_positives / nombre_total_predictions) * 100
        print(f"Pourcentage de prédictions positives : {pourcentage_predictions_positives:.2f}%")
    else:
        print("Fichier introuvable.")