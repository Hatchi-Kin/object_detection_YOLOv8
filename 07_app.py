from flask import Flask, render_template, flash, redirect, request
from ultralytics import YOLO
from PIL import Image
import cv2
import time


app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'

model = YOLO("best_model/weights/best.pt", task='detect')


def detect_objects_on_image(buf, model):

    results = model.predict(buf)
    result = results[0]
    output = []
    image = buf.copy()

    for box in result.boxes:

        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]

        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([x1, y1, x2, y2, result.names[class_id], prob])

        label = f"{result.names[class_id]}: {prob}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(image, label, (x1 + 6, y1 + label_size[1] + label_size[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return output, image


###################################################

@app.route('/')
def index():
    return render_template('upload.html')

###################################################

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        flash('No image uploaded')
        return redirect(request.url)
    
    image_file = request.files['image']
    image = cv2.imread(image_file)
    
    boxes, image_with_boxes = detect_objects_on_image(image, model)

    image_with_boxes = Image.fromarray(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
    image_with_boxes.save('static/image_with_boxes.jpg')
    
    return render_template('results.html', objects=boxes, image_file='image_with_boxes.jpg')

###################################################

import time


@app.route('/webcam')
def webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        boxes, image_with_boxes = detect_objects_on_image(frame, model)

        # Display the frame with detected objects
        cv2.imshow('frame', image_with_boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return render_template('upload.html')


###################################################


if __name__ == '__main__':
    app.run(debug=True)