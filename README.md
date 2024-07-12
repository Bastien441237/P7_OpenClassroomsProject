# 🐶 Détection de Chiens avec YOLOV9 et VGG16

Bienvenue dans le repository du projet **Détection de Chiens avec YOLOV9 et VGG16**. Ce projet utilise des modèles de deep learning pour détecter et classifier des images de chiens. Les images proviennent du Stanford Dogs Dataset et d'autres sources.

## 📚 Contexte du Projet

Réaliser une preuve de concept avec le modèle YOLOV9, nettoyage des images avec détection des chiens (cropping) puis réentraînement du modèle pour différencier les races de chiens. L'idée est de réussire à prouver qu'un modèle de détection d'objet peut être meilleur qu'un modèle de classification dans la prédiction de races de chiens.  

## 🎯 Objectifs du Projet

1. **Détecter les chiens dans les images à l'aide du modèle YOLOV9 et les isoler (cropping) afin de nettoyer la donnée.**
2. **Réentrainer YOLOV9 sur les données nettoyées**
3. **Classifier les races de chiens détectées à l'aide du modèle VGG16.**
4. **Comparer les deux modèles**
5. **Tester et valider les modèles sur des ensembles de données spécifiques.**
6. **Développer un dashboard interactif avec Streamlit pour afficher les résultats.**

## 📦 Livrables

1. **Un plan de travail prévisionnel** pour expliquer les choix dans la preuve de concept.
2. **Un modèle YOLOV9** pour la détection des chiens.
2. **Un modèle VGG16** pour la classification des races de chiens.
3. **Un notebook Jupyter** contenant les scripts de preprocessing, de détection, et de classification.
4. **Un dashboard Streamlit** pour la visualisation interactive des résultats.
5. **Une présentation** résumant les méthodes et les résultats du projet.

## 📂 Structure du Repository

```plaintext
├── Data_test/                                     # Données de test
├── Images/                                        # Images du Stanford Dogs Dataset
├── Images_streamlit/                              # Images pour l'application Streamlit
├── Images_test/                                   # Banque d'images pour tester le modèle
├── dogs_cutout_15/                                # Images de chiens identifiées et détourées par YOLOV9
├── dogs_detection_15/                             # Images de chiens détectées par YOLOV9
├── models/                                        # Modèles YOLOV9
├── runs/detect/exp/                               # Modèle YOLOV9
├── utils/                                         # Utilitaires pour le modèle YOLOV9
├── Moreno_Bastien_1_plan_travail_052024.pdf       # Livrable écrit : plan de travail
├── Moreno_Bastien_2_notebook_052024.ipynb         # Notebook du projet
├── Moreno_Bastien_3_note_methodo_052024.pdf       # Livrable écrit : note méthodologique
├── Moreno_Bastien_4_code_dashboard_052024.py      # Code du Dashboard Streamlit
├── Moreno_Bastien_5_presentation_052024.pdf       # Présentation
├── best.pt                                        # Modèle YOLOV9 réentrainé
├── export.py                                      # Script d'exportation du modèle YOLOV9
├── index_to_class.pkl                             # Dictionnaire des labels pour VGG16
├── model_vgg16.h5                                 # Modèle VGG16
├── n02088094_294.jpg                              # Image utilisée pour l'inférence
├── packages.txt                                   # Liste des packages
├── requirements.txt                               # Fichier des dépendances
└── README.md                                      # Ce fichier
```

## ⚙️ Instructions pour l'Installation
1. Clonez le dépôt GitHub :
```
git clone https://github.com/Bastien441237/P7_OpenClassroomsProject.git
```

2. Créez un environnement virtuel et activez-le :
```
python -m venv env
source env/bin/activate  # Pour Windows: env\Scripts\activate
```

3. Intaller les dépendances : 
```
pip install -r requirements.txt
```

## 🧑‍💻 Utilisation
### Détection avec YOLOV9
Pour détecter des chiens dans des images avec YOLOV9, utilisez le script export.py :
```
python export.py --source chemin_vers_images --weights best.pt --output chemin_vers_sortie
```

### Classification avec VGG16
Pour classifier des images de chiens avec VGG16, chargez les modèles et utilisez les scripts dans le notebook Moreno_Bastien_2_notebook_052024.ipynb.

### Dashboard Streamlit
Vous pouvez lancer le Dashboard en local : 
```
streamlit run Moreno_Bastien_4_code_dashboard_052024.py
```

ou bien la consulter en ligne : https://dashboard-yolov9.streamlit.app/

## 👨‍💻 Auteur
Bastien Moreno - Data Scientist et passionné par l'analyse de données et le développement de modèles intelligents.\
Pour en savoir plus sur moi et mes projets, n'hésitez pas à me contacter via mon [![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/bastien-moreno441237/).