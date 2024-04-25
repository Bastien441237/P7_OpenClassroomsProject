import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
from PIL import Image
import random

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
    selected_dirs = ['Bernese_mountain_dog', 'boxer', 'briard',
                    'Brittany_spaniel', 'bull_mastiff', 'Doberman',
                    'EntleBucher', 'French_bulldog', 'Gordon_setter',
                    'Greater_Swiss_Mountain_dog', 'Irish_setter', 'Rottweiler',
                    'Samoyed', 'Tibetan_terrier', 'vizsla']

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
    st.image(img, caption='Image', width=300)

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
