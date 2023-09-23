# Detection d'object avec YOLOv8

## Structure du projet
```
.
├─── dataset_from_roboflow.yolov8
│   ├─── test
│   ├─── train
│   └─── valid
│
├─── 01_scraping.py
├─── 02_data_harvest.py
├─── 03_resize.py
├─── 04_random_check.ipynb
├─── 05_train_yolov8.ipynb
├─── 06_test_display_results.py
│
├─── 07_app.py
├─── requirements.txt
│
├─── best_model
│   └─── weights
│       └─── best.pt
│
├─── static
│   ├─── css
│   ├─── image_with_boxes.jpg
│   ├─── test01.jpg
│   └─── test02.jpg
│
├─── templates
    ├─── results.html
    └─── upload.html
```

## Pour lancer l'application Flask ( dans un .venv ):
```
pip install requirements.txt
```

```
python 07_app.py
```