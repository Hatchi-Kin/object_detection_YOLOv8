# 📷 Detection d'object avec YOLOv8
🔭 pour plus d'info --> rapport.pdf
## 🌳  Structure du projet
```
OBJECT_DETECTION_YOLOV8
├── app.py
├── tree.txt
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
    ├── results.html
    └── upload.html
```

## 💾 La base de donnée est disponible sur roboflow:
 ```
https://universe.roboflow.com/yolosafetygear/safety_gear_simplon/dataset/3
 ```

## 💪 L'entrainement du modèle à été fait sur Google Colab:
 ```
 https://colab.research.google.com/drive/1haVoMxOHWRqEHvxCUKGXXVGnhcuMjPEo
 ```

## 🏃  Pour lancer l'application Flask ( dans un .venv ):

```
pip install requirements.txt
```

```
python app.py
```
