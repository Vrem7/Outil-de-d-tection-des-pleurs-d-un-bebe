import sounddevice as sd
from scipy.io.wavfile import write
import os
import numpy as np
import librosa
from keras.models import Sequential, load_model
import pygame
import RPi.GPIO as GPIO
import time

# Constantes
SEUIL_DETECTION_SON = 0.01
SEUIL_PLEURE = 60
SR = 44100
DUREE_ENREGISTREMENT = 6
TAILLE_SEGMENT = 80
ENREGISTREMENT_EN_COURS = False
DUREE_LECTURE_BERCEUSE = 60
SPEAKER_PIN = 3

# Configuration des broches GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SPEAKER_PIN, GPIO.OUT)

# Chemins de fichiers
DOSSIER_SAUVEGARDE = 'C:\\Users\\AskyM\\OneDrive\\Documents\\Projet_bebe\\Save'
DOSSIER_ENREGISTREMENT = 'C:\\Users\\AskyM\\OneDrive\\Documents\\Projet_bebe\\Son\\enregistrement'
DOSSIER_BERCEUSE = 'C:\\Users\\AskyM\\OneDrive\\Documents\\Projet_bebe\\Son'

NOM_MODELE = 'Train.h5'
NOM_ENREGISTREMENT = 'son.mp3'
NOM_BERCEUSE = 'berceuse.mp3'

CHEMIN_MODELE = os.path.join(DOSSIER_SAUVEGARDE, NOM_MODELE)
CHEMIN_ENREGISTREMENT = os.path.join(DOSSIER_ENREGISTREMENT, NOM_ENREGISTREMENT)
CHEMIN_BERCEUSE = os.path.join(DOSSIER_BERCEUSE, NOM_BERCEUSE)

pygame.init()
pygame.mixer.init()

modele = load_model(CHEMIN_MODELE)
print("Le modèle est chargé.")

print("En attente de détection de son...")

# Fonction de détection du son
def detection(indata, frames, time, status):
    volume_norm = abs(indata).max()
    global ENREGISTREMENT_EN_COURS

    if volume_norm > SEUIL_DETECTION_SON and not ENREGISTREMENT_EN_COURS:
        ENREGISTREMENT_EN_COURS = True
    elif volume_norm <= SEUIL_DETECTION_SON and ENREGISTREMENT_EN_COURS:
        ENREGISTREMENT_EN_COURS = False

# Fonction pour jouer la berceuse
def play_berceuse(file_path, duration):
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    time.sleep(duration)
    pygame.mixer.music.stop()

# Fonction pour obtenir les caractéristiques audio d'un fichier
def caracteristiques_audio(fichier_audio, taille_segment):
    donnees_audio, _ = librosa.load(fichier_audio, sr=SR)
    spectrogramme = librosa.feature.melspectrogram(y=donnees_audio, sr=SR)
    mfccs = librosa.feature.mfcc(y=donnees_audio, sr=SR, n_mfcc=13)

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

# Enregistrement et traitement du son
with sd.InputStream(callback=detection):
    try:
        while True:
            if ENREGISTREMENT_EN_COURS:
                print("Son détecté. Enregistrement en cours...")
                record = sd.rec(int(DUREE_ENREGISTREMENT * SR), samplerate=SR, channels=2)
                sd.wait()

                i = 0
                while os.path.exists(CHEMIN_ENREGISTREMENT):
                    i += 1
                    NOM_ENREGISTREMENT = f'son{i}.mp3'
                    CHEMIN_ENREGISTREMENT = os.path.join(DOSSIER_ENREGISTREMENT, NOM_ENREGISTREMENT)

                write(CHEMIN_ENREGISTREMENT, SR, record)
                print("Enregistrement terminé.")

                segments_spectrogramme_test, segments_mfccs_test = caracteristiques_audio(CHEMIN_ENREGISTREMENT, TAILLE_SEGMENT)
                segments_test = np.concatenate((np.array(segments_spectrogramme_test), np.array(segments_mfccs_test)), axis=1)
                predictions = modele.predict(np.array(segments_test))

                nombre_predictions_positives = np.sum(predictions)
                nombre_total_predictions = len(predictions)

                pourcentage_predictions_positives = (nombre_predictions_positives / nombre_total_predictions) * 100
                print(f"Pourcentage de prédictions positives : {pourcentage_predictions_positives:.2f}%")

                if pourcentage_predictions_positives > SEUIL_PLEURE:
                    play_berceuse(CHEMIN_BERCEUSE, DUREE_LECTURE_BERCEUSE)

    except KeyboardInterrupt:
        print("Arrêt manuel de l'enregistrement.")