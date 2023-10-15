# 📷 Detection d'object avec YOLOv8
🔭 pour plus d'info --> rapport.pdf

YOLOv8 est un modèle de détection d'objets populaire qui est connu pour sa vitesse et sa précision. Il est disponible dans plusieurs variantes, dont YOLOv8n, qui est une version plus légère et plus rapide.

Cette web app Flask utilise un YOLOV8n fine-tuné sur un custom dataset et permet de détecter la présence de gilets et de casques de sécurité sur des images. 👷

## 🌳  Structure du projet
```
OBJECT_DETECTION_YOLOV8
├── app.py
├── rapport.pdf
├── readme.md
├── requirements.txt
│
├── best_model
│   ├── confusion_matrix_normalized.png
│   └── weights
│       └── best.pt
├── model
│   ├── 01_scraping.py
│   ├── 02_data_harvest.py
│   ├── 03_resize.py
│   ├── 04_random_check.ipynb
│   ├── 05_train_yolov8.ipynb
│   └── 06_test_display_results.py
├── static
│   ├── image_with_boxes.jpg
│   ├── test01.jpg
│   └── test02.jpg
└── templates
    └── upload.html
```

## 💾 La base de données est disponible sur roboflow:
 ```
https://universe.roboflow.com/yolosafetygear/safety_gear_simplon/dataset/3
 ```

## 💪 L'entrainement du modèle à été fait sur Google Colab:
 ```
 https://colab.research.google.com/drive/1haVoMxOHWRqEHvxCUKGXXVGnhcuMjPEo
 ```

## 🏃  Pour lancer l'application Flask:

```
pip install requirements.txt
```

```
python app.py
```
