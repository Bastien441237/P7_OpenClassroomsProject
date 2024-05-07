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
import shutil

# Lien vers le Dashboard en ligne : https://bastien441237-p7--moreno-bastien-4-code-dashboard-052024-qnwzwh.streamlit.app/

# Fonction copiée du fichier detect.py (car impossible de lancer une ligne de commande sur streamlit en ligne)
import platform
import sys
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

@smart_inference_mode()
def run(
        weights=ROOT / 'yolo.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

######################################################################
# Téléchargement du modèle
import gdown

# Lien direct vers le fichier sur Google Drive
url = 'https://drive.google.com/uc?id=11fV815aVkGKY4jr0xik9AMl1zGHaEN3A'

# Emplacement où enregistrer le fichier téléchargé
destination = 'modelbest.pt'

# Téléchargement du fichier
gdown.download(url, destination, quiet=False)
########################################################################
# Chemin vers le répertoire d'images
images_dir = os.path.join('Images')

# Boucle pour récupérer les noms d'images et les labels
image_files = []
labels = []

for root, dirs, images in os.walk(images_dir):
    for image in images:
        image_files.append(os.path.join(root, image))
        labels.append(os.path.basename(root).split('-')[1])

# Mise en place du DataFrames
df_dogs = pd.DataFrame({'image_path': image_files, 'label': labels})

# Création de la sidebar
st.sidebar.title("Taille du texte :")
text_size = st.sidebar.select_slider("(en %)", options=list(range(50, 301)), value=100)

st.sidebar.title("Sommaire")
pages = ['Contexte du projet', 'Analyse exploratoire des données', 'Nettoyage des données', 'Choix du modèle', 'Prédiction du modèle']
page = st.sidebar.radio("Aller à la page :", pages)

# Chargement des modèles
model_path_vgg16 = 'model_vgg16.h5'
model_vgg16 = tf.keras.models.load_model(model_path_vgg16)
model_filename = './modelbest.pt'
model_path_yolov9 = os.path.join(model_filename)

# Chargement des labels
with open('index_to_class.pkl', 'rb') as file:
    classes_labels = pickle.load(file)

# Fonction pour charger et redimensionner les images
def load_and_resize_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    return img_array

# Dossier contenant les images de test par race de chiens
data_test_dir = os.path.join('./Data_test')

# Classes sélectionnées
selection_classes = ['Bernese_mountain_dog', 'boxer', 'briard',
                    'Brittany_spaniel', 'bull_mastiff', 'Doberman',
                    'EntleBucher', 'French_bulldog', 'Gordon_setter',
                    'Greater_Swiss_Mountain_dog', 'Irish_setter', 'Rottweiler',
                    'Samoyed', 'Tibetan_terrier', 'vizsla']

# Initialiser un dictionnaire pour stocker le nombre d'images par étiquette
label_counts = {}

# Boucle pour récupérer les noms d'images et les labels
for root, dirs, images in os.walk(images_dir):
    for image in images:
        label = os.path.basename(root).split('-')[1]
        label_counts[label] = label_counts.get(label, 0) + 1
df_label_counts = pd.DataFrame(label_counts.items(), columns=['Label', "Nombre d'images"])

# Afficher un describe
label_statistics = df_label_counts.describe()

# Ajouter une colonne pour le total d'images
label_statistics.loc['Total'] = df_label_counts["Nombre d'images"].sum()

# Application Streamlit
# Page d'explication du contexte du projet
if page == pages[0]:
    st.title('Contexte du projet')
    st.write("### YOLOv9, modèle de détection d'objets, peut-il améliorer la classification d'images dans la vision par ordinateur ?")

    # Chemin d'accès à l'image
    img = "./Images_streamlit/YoloV9.png"
    # Charger l'image à partir du chemin d'accès
    img = Image.open(img)
    # Afficher l'image dans Streamlit
    st.image(img, width=700)

    texte = "Dans le cadre de notre projet, nous avons entrepris d'explorer \
            les capacités de YOLOv9, une version récente et améliorée du \
            modèle YOLO (*You Only Look Once*), largement utilisé pour la détection \
            d'objets dans les images et les vidéos. YOLOv9 s'inscrit dans la lignée \
            de ses prédécesseurs, bénéficiant de plusieurs améliorations et optimisations\
            qui en font un choix attrayant pour notre étude. Disponible depuis quelques\
            mois seulement, ce modèle a déjà fait ses preuves en matière de détection\
            d'objets, y compris la détection de chiens dans des images et des flux vidéo. \
            Les retours et tests disponibles suggèrent que YOLOv9 offre une performance accrue, \
            notamment dans la détection d'objets en mouvement, ce qui en fait un choix prometteur \
            pour notre exploration dans le domaine spécifique de la classification des races de chiens."

    # Affichage du texte avec la taille sélectionnée
    st.markdown(f'<p style="font-size:{text_size}%">{texte}</p>', unsafe_allow_html=True)

elif page == pages[1]:
    st.title('Analyse exploratoire des données')

    texte = "Afin de tester le modèle YOLOv9, nous avons choisis d'utiliser le Stanford Dogs Dataset\
             qui contient 120 races différentes de chiens et 20 580 images."
    st.markdown(f'<p style="font-size:{text_size}%">{texte}</p>', unsafe_allow_html=True)
    
    st.write("### Affichage des statistiques")
    st.dataframe(label_statistics)
    texte = "On a donc 120 classes différentes avec en moyenne 171 images par classes."
    st.markdown(f'<p style="font-size:{text_size}%">{texte}</p>', unsafe_allow_html=True)

    st.write("### Nombre d'images par labels")
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

    # Section pour la sélection de données en entrée du moteur de prédiction
    st.write("### Visualisation d'images disponible dans le Stanford  Dogs Dataset")
    selected_labels = st.multiselect("Sélectionner les labels :", df_dogs['label'].unique())
    num_images_per_label = st.slider("Nombre d'images par labels :", 1, 10, 3)

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
        texte = "Aucune étiquette sélectionnée. Veuillez en sélectionner au moins une dans la barre latérale."
        st.markdown(f'<p style="font-size:{text_size}%">{texte}</p>', unsafe_allow_html=True)

elif page == pages[2]:
    # Titre de la page
    st.title('Nettoyage des données')

    texte = "Pour commencer, nous allons sélectionner uniquement 15 races de chiens, \
             ce qui permettra de réduire significativement le temps de calcul nécessaire. \
             Parmi ces 15 races, nous choisirons délibérément des races similaires où les \
             modèles rencontrent souvent des difficultés. L'objectif est d'obtenir un \
             échantillon représentatif des performances que nous pourrions atteindre sur \
             l'ensemble du Stanford Dogs Dataset."
    st.markdown(f'<p style="font-size:{text_size}%">{texte}</p>', unsafe_allow_html=True)
    
    # Liste des étiquettes sélectionnées
    selected_dirs = selection_classes

    # Section pour la sélection des échantillons de races de chiens
    st.write("### Visualisation d'images sur les 15 races de chiens")
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
        texte = "Aucune étiquette sélectionnée. Veuillez en sélectionner au moins une dans la barre latérale."
        st.markdown(f'<p style="font-size:{text_size}%">{texte}</p>', unsafe_allow_html=True)

    # Section sur l'explication de l'utilisation de YOLOV9 dans le nettoyage du Dataset
    st.write("### Identification des chiens avec YOLOV9")
    texte = "Nous avons ensuite utilisé YOLOv9 pour déterminer la présence de chiens dans les images. \
             En effet, celles-ci peuvent contenir non seulement des chiens, mais aussi des humains, \
             d'autres animaux ou même des objets."
    st.markdown(f'<p style="font-size:{text_size}%">{texte}</p>', unsafe_allow_html=True)
    # Chemin d'accès de l'image
    image_path = "n02088094_294.jpg"
    # Charger l'image à partir du chemin d'accès
    img = Image.open(image_path)
    # Afficher l'image dans Streamlit
    st.image(img, width=300)

    # Affichage des images originales
    texte = "Ensuite, nous avons utilisé ces détections pour conserver uniquement les régions \
             contenant des chiens dans les images. L'objectif était de recadrer les images afin \
             de focaliser l'attention sur les éléments les plus pertinents pour la classification \
             des races de chiens. Ci-dessous, vous pouvez voir les images une fois recadrées :"
    st.markdown(f'<p style="font-size:{text_size}%">{texte}</p>', unsafe_allow_html=True)
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

    # Section sur l'explication de l'utilisation de RMBG 1.4 de BRIA AI dans le nettoyage du Dataset
    st.write("### Réduction du bruit via RMBG 1.4 de BRIA AI")

    # Affichage des images détourées
    texte = "Enfin, afin de réduire au maximum le bruit dans les images, nous avons opté pour \
             l'utilisation du modèle de Bria AI (RMBG 1.4), qui offre la possibilité de détourer l'image. \
             Vous pouvez voir ci-dessous les images recadrées et détourées :"
    st.markdown(f'<p style="font-size:{text_size}%">{texte}</p>', unsafe_allow_html=True)
    
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
    texte = "Pour notre modèle de base, nous avons opté pour le modèle VGG16, \
             qui a été préalablement entraîné sur les données d'ImageNet. \
             Notre ensemble de données est également issu d'ImageNet, \
             ce qui constitue une base solide pour la classification des \
             différentes races de chiens. Nous avons entraîné ce modèle \
             pré-entraîné sur les images brutes, c'est-à-dire non nettoyées. \
             Mais nous avons utilisé les couches d'input de VGG16 pour \
             le preprocessing de celles-ci."
    st.markdown(f'<p style="font-size:{text_size}%">{texte}</p>', unsafe_allow_html=True)
    
    # Chemin d'accès de l'évaluation VGG16
    img1 = "./Images_streamlit/VGG16_evaluate.png"

    # Charger l'image à partir du chemin d'accès
    img = Image.open(img1)
    st.write('#### Evaluation du modèle')

    # Afficher l'image dans Streamlit
    st.image(img, width=1000)
    st.write('#### Matrice de confusion sur les données de test')

    # Chemin d'accès de la matrice de confusion VGG16
    img2 = "./Images_streamlit/VGG16_confusion_matrix.png"

    # Charger l'image à partir du chemin d'accès
    img = Image.open(img2)

    # Afficher l'image dans Streamlit
    st.image(img, width=800)

    # Explication des résultats
    texte = "Lors de son entraînement, VGG16 a tendance à présenter des signes de surapprentissage, \
             même avec l'utilisation d'un EarlyStopping pour limiter le nombre d'epochs. Cependant, \
             cette tendance est mieux comprise lorsque nous examinons les scores de la matrice de confusion. \
             Comme prévu avec l'échantillonnage des 15 races de chiens, le modèle éprouve des difficultés \
             à distinguer les races similaires."
    st.markdown(f'<p style="font-size:{text_size}%">{texte}</p>', unsafe_allow_html=True)
    texte = "Cette baseline nous fournit un premier score à partir duquel nous allons pouvoir évaluer les \
             performances de YOLOv9, ainsi que l'impact de notre processus de nettoyage préalable."
    st.markdown(f'<p style="font-size:{text_size}%">{texte}</p>', unsafe_allow_html=True)

    # Modèle YOLOV9 sur les images non détourées
    st.write('### Modèle YOLOv9 sur les images non détourées')
    st.write('#### Evaluation du modèle')

    # Chemin d'accès de l'évaluation YOLOV9
    img3 = "./Images_streamlit/YOLOV9_results_1.png"

    # Charger l'image à partir du chemin d'accès
    img = Image.open(img3)

    # Afficher l'image dans Streamlit
    st.image(img, width=900)
    st.write('#### Matrice de confusion sur les données de test')

    # Chemin d'accès de la matrice de confusion YOLOV9
    img4 = "./Images_streamlit/YOLOV9_confusion_matrix_1.png"

    # Charger l'image à partir du chemin d'accès
    img = Image.open(img4)

    # Afficher l'image dans Streamlit
    st.image(img, width=1000)

    # Explication des résultats
    texte = "Le modèle YOLOv9 a été entraîné sur 70 epochs en utilisant les images non détourées, \
             aboutissant à une accuracy de 0.63. Cette performance est inférieure à celle du modèle VGG16. \
             Cependant, une analyse plus approfondie de la matrice \
             de confusion révèle que le modèle commence à discerner les races de chiens, bien qu'il \
             éprouve encore des difficultés avec les races similaires."
    st.markdown(f'<p style="font-size:{text_size}%">{texte}</p>', unsafe_allow_html=True)

    # Modèle YOLOV9 sur les images détourées
    st.write('### Modèle YOLOV9 sur les images détourées')
    st.write('#### Evaluation du modèle')

    # Chemin d'accès de l'évaluation YOLOV9
    img5 = "./Images_streamlit/YOLOV9_results_2.png"

    # Charger l'image à partir du chemin d'accès
    img = Image.open(img5)

    # Afficher l'image dans Streamlit
    st.image(img, width=900)
    st.write('#### Matrice de confusion sur les données de test')

    # Chemin d'accès de la matrice de confusion YOLOV9
    img6 = "./Images_streamlit/YOLOV9_confusion_matrix_2.png"

    # Charger l'image à partir du chemin d'accès
    img = Image.open(img6)

    # Afficher l'image dans Streamlit
    st.image(img, width=1000)

        # Explication des résultats
    texte = "Ensuite, nous avons entraîné le modèle YOLOv9 sur les images détourées, \
             ce qui a donné des résultats améliorés sur le même nombre d'epochs (70). \
             Cependant, nous avons augmenté le nombre d'epochs à 100 pour tenter de faire en sorte que \
             le modèle puisse s'améliorer sur la classification des races de chiens similaires."
    st.markdown(f'<p style="font-size:{text_size}%">{texte}</p>', unsafe_allow_html=True)
    texte = "Cela nous a permis d'atteindre une accuracy de 0.73. De plus, l'analyse de la matrice de confusion révèle une \
             amélioration dans la reconnaissance des races de chiens similaires."
    st.markdown(f'<p style="font-size:{text_size}%">{texte}</p>', unsafe_allow_html=True)

    st.write('### Conclusion')
    texte = "Le modèle YOLOv9 entraîné sur les images détourées permet d'obtenir le meilleur score. \
             Nous allons donc choisir ce modèle."
    st.markdown(f'<p style="font-size:{text_size}%">{texte}</p>', unsafe_allow_html=True)
    
    # Liste des races de chiens disponibles
    dog_breeds = os.listdir(data_test_dir)

    # Titre pour les paramètres de prédiction test
    st.sidebar.title("Choix des paramètres de prédiction test :")

    # Sidebar pour choisir la race de chien
    selected_breed = st.sidebar.selectbox("Choisir une race de chien :", dog_breeds)

    # Sidebar pour choisir le nombre d'images par race de chiens (1 à 2)
    num_images_per_breed = st.sidebar.slider("Nombre d'images par race de chiens :", 1, 2, 1)

    # Titre pour les prédictions
    st.write('### Résultats des prédictions sur les données de test')

    # Bouton pour déclencher les prédictions
    if st.sidebar.button('Prédire'):
        exp = 1
        # Créer un conteneur vide pour afficher le contenu en fonction de l'état du spinner
        container = st.empty()
        with st.spinner('Prédictions en cours...'):
            st.write(f"##### Race de chien : {selected_breed}")
            breed_dir = os.path.join(data_test_dir, selected_breed)
            if os.path.isdir(breed_dir):
                breed_images = os.listdir(breed_dir)
                random.shuffle(breed_images)
                images_selected = breed_images[:min(num_images_per_breed, len(breed_images))]
                for image_name in images_selected:
                    image_path = os.path.join(breed_dir, image_name)
                    image_yolo_path = os.path.join('./runs', 'detect', f'exp{exp+1}', image_name)
                    
                    # Charger l'image
                    with open(image_path, 'rb') as f:
                        image = Image.open(f)
                    
                    # Redimensionner et normaliser l'image
                    img = kimage.load_img(image_path, target_size=(224, 224))
                    img = kimage.img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                    img = img / 255.0

                    # Utiliser le modèle YOLOV9 avec la fonction run
                    run(weights=model_path_yolov9, source=image_path, device='cpu')

                    # Faire des prédictions avec le modèle VGG16
                    prediction = model_vgg16.predict(img)
                    
                    # Obtenir la classe prédite et la probabilité du modèle VGG16
                    max_index = np.argmax(prediction)
                    max_label = classes_labels[max_index]
                    max_proba = prediction[0][max_index]*100
                    
                    # Afficher les prédictions
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Prédiction VGG16**")
                        with open(image_path, 'rb') as f:
                            image = Image.open(f)
                            st.image(image, width=150)
                            st.write(f"**{max_label}** à **{max_proba:.2f}%**")
                    with col2:
                        st.write("**Prédiction YOLOV9**")
                        with open(image_yolo_path, 'rb') as f:
                            image2 = Image.open(f)
                            st.image(image2, width=150)
                    exp = exp + 1

        # Chemin du dossier contenant les dossiers exp
        exp_folder = os.path.join('./runs', 'detect')

        # Liste de tous les dossiers exp présents dans le répertoire
        exp_folders = [folder for folder in os.listdir(exp_folder) if folder.startswith('exp')]
        for exp_dir in exp_folders:
            if exp_dir != 'exp':
                exp_dir_path = os.path.join(exp_folder, exp_dir)
                shutil.rmtree(exp_dir_path)
    else:
        texte = "<i>Veuillez sélectionner les paramètres sur la barre latérale gauche,\
                     puis cliquer sur '<B>Prédire</B>' afin d'afficher les prédictions.<i>"
        st.markdown(f'<p style="font-size:{text_size}%">{texte}</p>', unsafe_allow_html=True)

elif page == pages[4]:
    st.title("Prédiction du modèle")
    texte = "Vous pouvez choisir une image dans la banque d'images mise à disposition \
             ou bien uploader directement une image pour tester le modèle YOLOv9."
    st.markdown(f'<p style="font-size:{text_size}%">{texte}</p>', unsafe_allow_html=True)

    images_dir = os.path.join('./Images_test')

    image_files = os.listdir(images_dir)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Choix d'une image dans la banque d'image :")
        selected_image = st.selectbox("Sélectionner une image :", image_files, index=0)
        image_path = os.path.join(images_dir, selected_image)
        image = Image.open(image_path)
        image_placeholder = st.empty()
        image_placeholder.image(image, caption=selected_image, use_column_width=True)
        
        if st.button("Prédire avec YOLOV9"): 
            with st.spinner('Prédiction en cours...'):
                # Redimensionner et normaliser l'image
                img = kimage.load_img(image_path, target_size=(224, 224))
                img = kimage.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = img / 255.0

                # Utiliser le modèle YOLOV9
                run(weights=model_path_yolov9, source=image_path, device='cpu')

                # Afficher l'image avec la boîte de prédiction
                image_yolo_path = os.path.join('./runs', 'detect', 'exp2', selected_image)
                with open(image_yolo_path, 'rb') as f:
                    image2 = Image.open(f)
                    image_placeholder.image(image2, caption="Prédiction YOLOV9", use_column_width=True)

            # Chemin du dossier contenant les dossiers exp
            exp_folder = os.path.join('./runs', 'detect')
            
            # Supprimer le dossier exp2
            exp_folders = [folder for folder in os.listdir(exp_folder) if folder.startswith('exp')]
            for exp_dir in exp_folders:
                if exp_dir != 'exp':
                    exp_dir_path = os.path.join(exp_folder, exp_dir)
                    shutil.rmtree(exp_dir_path)
    with col2:
        st.subheader("Veuillez charger votre image ci-dessous :")
        upload = st.file_uploader("Charger l'image du chien :", type=['png', 'jpeg', 'jpg'])
        if upload:
            # Enregistrer l'image téléchargée dans un dossier créé à cet effet
            save_folder = os.path.join('./runs', 'detect', 'exp2')
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, upload.name)
            with open(save_path, 'wb') as f:
                f.write(upload.read())

            with st.spinner('Prédiction en cours...'):
                # Redimensionner et normaliser l'image
                img = kimage.load_img(save_path, target_size=(224, 224))
                img = kimage.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = img / 255.0

                # Utiliser le modèle YOLOV9
                run(weights=model_path_yolov9, source=save_path, device='cpu')

                # Afficher l'image avec la boîte de prédiction
                image_yolo_path = os.path.join('./runs', 'detect', 'exp3', upload.name)
                with open(image_yolo_path, 'rb') as f:
                    image2 = Image.open(f)
                    image_placeholder = st.empty()
                    image_placeholder.image(image2, caption="Prédiction YOLOV9", use_column_width=True)

            # Chemin du dossier contenant les dossiers exp
            exp_folder = os.path.join('./runs', 'detect')
            
            # Supprimer le dossier exp2
            exp_folders = [folder for folder in os.listdir(exp_folder) if folder.startswith('exp')]
            for exp_dir in exp_folders:
                if exp_dir != 'exp':
                    exp_dir_path = os.path.join(exp_folder, exp_dir)
                    shutil.rmtree(exp_dir_path)