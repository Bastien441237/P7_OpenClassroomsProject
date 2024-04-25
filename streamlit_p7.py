import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
from PIL import Image
import random
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing import image as kimage
import pickle
import numpy as np
import subprocess

# Chemin vers le répertoire d'images
# images_dir = "C:\\Users\\Basti\\Projets Python\\Machine Learning Engineer\\P7\\Images"
images_dir = ".\Images"

# Boucle pour récupérer les noms d'images et les labels
image_files = []
labels = []

for root, dirs, images in os.walk(images_dir):
    for image in images:
        image_files.append(os.path.join(root, image))
        labels.append(os.path.basename(root).split('-')[1])

# Mise en place des DataFrames
df_dogs = pd.DataFrame({'image_path': image_files, 'label': labels})
sample_df = df_dogs.sample(10)

st.sidebar.title("Sommaire")
pages = ['Contexte du projet', 'Analyse exploratoire des données', 'Nettoyage des données', 'Choix du modèle', 'Prédiction du modèle']
page = st.sidebar.radio("Aller à la page :", pages)

# Chargement des modèles
model_path_vgg16 = 'model_vgg16.h5'
model_vgg16 = tf.keras.models.load_model(model_path_vgg16)

model_dir = 'yolov9_model'
model_filename = 'best.pt'
model_path_yolov9 = os.path.join(model_dir, model_filename)

# Chargement des labels
with open('index_to_class.pkl', 'rb') as file:
    classes_labels = pickle.load(file)

# Fonction pour charger et redimensionner les images
def load_and_resize_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    return img_array

# Dossier contenant les images de test par race de chiens
data_test_dir = 'Data_test'

# Classes sélectionnées
selection_classes = ['Bernese_mountain_dog', 'boxer', 'briard',
                    'Brittany_spaniel', 'bull_mastiff', 'Doberman',
                    'EntleBucher', 'French_bulldog', 'Gordon_setter',
                    'Greater_Swiss_Mountain_dog', 'Irish_setter', 'Rottweiler',
                    'Samoyed', 'Tibetan_terrier', 'vizsla']

if page == pages[0]:
    st.title("Contexte du projet")

elif page == pages[1]:
    st.title('Analyse exploratoire des données')

    # Grouper et compter le nombre d'images par étiquette
    nb = df_dogs.groupby("label").count().sort_values(by="image_path", ascending=False)

    # Tracer le graphique
    plt.figure(figsize=(12, 25))
    sns.set(style="whitegrid")
    ax = sns.barplot(x="image_path", y=nb.index, data=nb)
    plt.xlabel("Nombre d'images")
    plt.ylabel('Labels')
    plt.title("Nombre d'images par label", fontdict={'fontweight':'bold', 'fontsize':15})
    plt.tight_layout()
    st.pyplot(plt)

    # Afficher les images et leurs étiquettes
    st.write("#### Images aléatoires et leurs étiquettes")
    plt.figure(figsize=(12, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        img = mpimg.imread(sample_df.iloc[i]['image_path'])
        plt.imshow(img)
        plt.title(sample_df.iloc[i]['label'], fontdict={'fontweight':'bold'})
        plt.axis('off')  # Supprimer les axes
    plt.tight_layout()
    st.pyplot(plt)

    # Section pour la sélection de données en entrée du moteur de prédiction
    st.write("## Sélection de données en entrée")
    selected_labels = st.multiselect("Sélectionner les étiquettes :", df_dogs['label'].unique())
    num_images_per_label = st.slider("Nombre d'images par étiquette :", 1, 10, 3)

    # Filtrer le DataFrame en fonction des étiquettes sélectionnées
    filtered_df = df_dogs[df_dogs['label'].isin(selected_labels)]

    # Afficher les images et leurs étiquettes
    st.write("Images sélectionnées :")
    if len(selected_labels) > 0:
        for label in selected_labels:
            st.write(f"Images pour l'étiquette '{label}'")
            images_for_label = filtered_df[filtered_df['label'] == label].sample(num_images_per_label)
            plt.figure(figsize=(12, 5))
            for i, (_, row) in enumerate(images_for_label.iterrows(), 1):
                plt.subplot(1, num_images_per_label, i)
                img = Image.open(row['image_path'])
                plt.imshow(img)
                plt.title(label, fontdict={'fontweight':'bold'})
                plt.axis('off')  # Supprimer les axes
            plt.tight_layout()
            st.pyplot(plt)  # Afficher le graphique dans Streamlit
    else:
        st.write("Aucune étiquette sélectionnée. Veuillez en sélectionner au moins une dans la barre latérale.")

elif page == pages[2]:
    # Titre de la page
    st.title('Nettoyage des données')
    st.write("Nous allons d'abord choisir 15 races de chiens afin de réduire le temps de calcul.")
    # Liste des étiquettes sélectionnées
    selected_dirs = selection_classes

    # Section pour la sélection de données en entrée du moteur de prédiction
    st.write("## Sélection de données en entrée")
    selected_labels = st.multiselect("Sélectionner les étiquettes :", selected_dirs)
    num_images_per_label = st.slider("Nombre d'images par étiquette :", 1, 10, 3)

    # Filtrer le DataFrame en fonction des étiquettes sélectionnées
    filtered_df = df_dogs[df_dogs['label'].isin(selected_labels)]

    # Afficher les images et leurs étiquettes
    st.write("Images sélectionnées :")
    if len(selected_labels) > 0:
        for label in selected_labels:
            st.write(f"Images pour l'étiquette '{label}'")
            images_for_label = filtered_df[filtered_df['label'] == label].sample(num_images_per_label)
            plt.figure(figsize=(12, 5))
            for i, (_, row) in enumerate(images_for_label.iterrows(), 1):
                plt.subplot(1, num_images_per_label, i)
                img = Image.open(row['image_path'])
                plt.imshow(img)
                plt.title(label, fontdict={'fontweight':'bold'})
                plt.axis('off')  # Supprimer les axes
            plt.tight_layout()
            st.pyplot(plt)  # Afficher le graphique dans Streamlit
    else:
        st.write("Aucune étiquette sélectionnée. Veuillez en sélectionner au moins une dans la barre latérale.")

    st.write("Nous avons ensuite utilisé YOLOV9 afin de pouvoir identifier si des chiens étaient présents sur les images.\
              En effet, il est possible d'avoir des humains, d'autres animaux ou bien même des objets sur les photos.")
    # Chemin d'accès de l'image
    image_path = "n02088094_294.jpg"
    # Charger l'image à partir du chemin d'accès
    img = Image.open(image_path)
    # Afficher l'image dans Streamlit
    st.image(img, width=300)

    # Affichage des images originales
    st.write("Puis, nous avons utilisé ces détections pour ne garder que les détections de chiens dans les images. \
                Le but est de recadrer les images sur ce qui va apporter le plus d'information à la classification par races de chiens. \
                Ci-dessous, les photos une fois recadrées :")
    # Chemin du dossier contenant les images détectées
    images_detect_dir = "./dogs_detection_15"

    # Liste des chemins d'accès des images originales
    image_paths_original = []
    for subdir in os.listdir(images_detect_dir):
        subdir_path = os.path.join(images_detect_dir, subdir)
        if os.path.isdir(subdir_path):
            image_paths_original.extend([os.path.join(subdir_path, img) for img in os.listdir(subdir_path) if img.endswith('.jpg')])

    # Sélectionner aléatoirement un sous-ensemble d'images
    random.shuffle(image_paths_original)
    num_images_to_display = 6
    image_paths_original = image_paths_original[:num_images_to_display]

    # Afficher les images originales dans un panel
    with st.container():  # Utiliser un conteneur pour organiser les images en colonnes
        col1, col2, col3 = st.columns(3)  # Créer trois colonnes
        for index, image_path in enumerate(image_paths_original):
            if index % 3 == 0:
                img = Image.open(image_path)
                with col1:
                    st.image(img, caption=f"Image {index+1}", width=150)
            elif index % 3 == 1:
                img = Image.open(image_path)
                with col2:
                    st.image(img, caption=f"Image {index+1}", width=150)
            else:
                img = Image.open(image_path)
                with col3:
                    st.image(img, caption=f"Image {index+1}", width=150)

    # Affichage des images détourées
    st.write("Puis, pour enlever un maximum de bruit sur l'image, nous avons choisis d'utiliser le modèle de Bria AI (RMBG 1.4) \
            qui permet de détourer l'image. Voici quelques images après traitement :")
    # Chemin du dossier contenant les images détourées
    images_cutout_dir = "./dogs_cutout_15"

    # Liste des chemins d'accès des images détourées avec les mêmes noms que les images originales
    image_paths_cutout = [img.replace(images_detect_dir, images_cutout_dir).replace('.jpg', '_RMBG.png') for img in image_paths_original]

    # Afficher les images détourées dans un panel
    with st.container():  # Utiliser un conteneur pour organiser les images en colonnes
        col1, col2, col3 = st.columns(3)  # Créer trois colonnes
        for index, image_path in enumerate(image_paths_cutout):
            if index % 3 == 0:
                img = Image.open(image_path)
                with col1:
                    st.image(img, caption=f"Image {index+1}", width=150)
            elif index % 3 == 1:
                img = Image.open(image_path)
                with col2:
                    st.image(img, caption=f"Image {index+1}", width=150)
            else:
                img = Image.open(image_path)
                with col3:
                    st.image(img, caption=f"Image {index+1}", width=150)

elif page == pages[3]:
    # Titre de la page
    st.title('Choix du modèle')
    st.write('### Modèle VGG16 baseline')
    st.write("Pour le modèle baseline nous avons choisi le modèle VGG16 qui a été entraîné sur les données d'ImageNet. \
             Ici, notre DataSet est composé de données d'ImageNet, cela permettra donc d'avoir une bonne base pour la classification \
             des images de races de chiens. Nous avons entraîné ce modèle préentraîné sur les images brutes, c'est-à-dire \
             non nettoyées.")
    # Chemin d'accès de l'évaluation VGG16
    img1 = "./Images_streamlit/VGG16_evaluate.png"
    # Charger l'image à partir du chemin d'accès
    img = Image.open(img1)
    st.write('#### Evaluation du modèle')
    # Afficher l'image dans Streamlit
    st.image(img, width=800)
    st.write('#### Matrice de confusion sur les données de test')
    # Chemin d'accès de la matrice de confusion VGG16
    img2 = "./Images_streamlit/VGG16_confusion_matrix.png"
    # Charger l'image à partir du chemin d'accès
    img = Image.open(img2)
    # Afficher l'image dans Streamlit
    st.image(img, width=800)
    # Modèle YOLOV9 sur les images non détourées
    st.write('### Modèle YOLOV9 sur les images non détourées')
    st.write('#### Evaluation du modèle')
    # Chemin d'accès de l'évaluation YOLOV9
    img3 = "./Images_streamlit/YOLOV9_results_1.png"
    # Charger l'image à partir du chemin d'accès
    img = Image.open(img3)
    # Afficher l'image dans Streamlit
    st.image(img, width=800)
    st.write('#### Matrice de confusion sur les données de test')
    # Chemin d'accès de la matrice de confusion YOLOV9
    img4 = "./Images_streamlit/YOLOV9_confusion_matrix_1.png"
    # Charger l'image à partir du chemin d'accès
    img = Image.open(img4)
    # Afficher l'image dans Streamlit
    st.image(img, width=900)
    # Modèle YOLOV9 sur les images détourées
    st.write('### Modèle YOLOV9 sur les images détourées')
    st.write('#### Evaluation du modèle')
    # Chemin d'accès de l'évaluation YOLOV9
    img5 = "./Images_streamlit/YOLOV9_results_2.png"
    # Charger l'image à partir du chemin d'accès
    img = Image.open(img5)
    # Afficher l'image dans Streamlit
    st.image(img, width=800)
    st.write('#### Matrice de confusion sur les données de test')
    # Chemin d'accès de la matrice de confusion YOLOV9
    img6 = "./Images_streamlit/YOLOV9_confusion_matrix_2.png"
    # Charger l'image à partir du chemin d'accès
    img = Image.open(img6)
    # Afficher l'image dans Streamlit
    st.image(img, width=900)
    st.write('### Conclusion')
    st.write("Le modèle YOLOV9 entraîné sur les images détourées permet d'obtenir le meilleur score. \
             Il est également intéressant de noter que ce modèle obtient de meilleurs résultats sur les données de test.")

    # Liste des races de chiens disponibles
    dog_breeds = os.listdir(data_test_dir)

    # Sidebar pour choisir le nombre d'images par race de chiens
    num_images_per_breed = st.sidebar.selectbox("Nombre d'images par race de chiens :", [1, 2, 3, 4, 5])

    # Bouton pour déclencher les prédictions
    if st.sidebar.button('Prédire'):
        for breed in dog_breeds:
            st.write(f"##### Race de chien : {breed}")
            breed_dir = os.path.join(data_test_dir, breed)
            if os.path.isdir(breed_dir):
                breed_images = os.listdir(breed_dir)[:num_images_per_breed]
                for image_name in breed_images:
                    image_path = os.path.join(breed_dir, image_name)
                    
                    # Charger l'image
                    with open(image_path, 'rb') as f:
                        image = Image.open(f)
                    
                    # Charger, redimensionner et normaliser l'image
                    img = kimage.load_img(image_path, target_size=(224, 224))
                    img = kimage.img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                    img = img / 255.0

                    ## Utiliser le modèle YOLOV9
                    # detect_yolov9 = '.\yolov9_model\detect.py'
                    # yolo_command = f"python detect.py --img 640 --device cpu --weights {model_path_yolov9} --source {image_path} --nosave --save-txt"
                    # yolo_output = subprocess.check_output(yolo_command, shell=True, text=True)
                    
                    # Faire des prédictions avec le modèle VGG16
                    prediction = model_vgg16.predict(img)
                    
                    # Obtenir la classe prédite et la probabilité du modèle VGG16
                    max_index = np.argmax(prediction)
                    max_label = classes_labels[max_index]
                    max_proba = prediction[0][max_index]
                    
                    # Afficher la prédiction et la probabilité du modèle VGG16
                    col1, col2 = st.columns(2)
                    with col1:
                        with open(image_path, 'rb') as f:
                            image = Image.open(f)
                            st.image(image, width=150)
                            st.write("Prédiction YOLOv9:")
                            st.write("AR")  # Afficher la prédiction de YOLOv9
                    with col2:
                        st.write(f"Race prédite par le modèle VGG16 **{max_label}** avec une probabilité de : **{max_proba:.2f}**")

elif page == pages[4]:
    st.title("Prédiction du modèle")