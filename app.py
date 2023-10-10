from flask import Flask, Response, render_template, request
from ultralytics import YOLO
from PIL import Image
import cv2


app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'

model = YOLO("best_model/weights/best.pt", task='detect')

###################################################

def draw_bounding_boxes(image, result):
    output = []
    for box in result.boxes:
        prob = round(box.conf[0].item(), 2)
        if prob > 0.6:
            x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
            class_id = box.cls[0].item()
            obj_label = result.names[class_id].strip()
            output.append([x1, y1, x2, y2, obj_label, prob])
            label = f"{obj_label}: {prob}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(image, label,
                        (x1 + 6, y1 + label_size[1] + label_size[1] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 185, 0), 3)
    return output


def increment_object_counts(obj_count, output):
    for obj in output:
        obj_label = obj[4]
        if obj_label not in obj_count:
            obj_count[obj_label] = 0
        obj_count[obj_label] += 1
    return obj_count


def check_equipment_completion(obj_count):
    if '0 personne' in obj_count:
        personne_count = obj_count['0 personne']
        gilet_count = obj_count.get('2 gilet', 0)
        casque_count = obj_count.get('1 casque', 0)
        if personne_count > gilet_count + 2 * casque_count:
            label = 'equipment incomplet'
            return label
    return None


def detect_objects_on_image(buf, model):
    results = model.predict(buf)
    result = results[0]
    image = buf.copy()
    output = draw_bounding_boxes(image, result)
    obj_count = increment_object_counts(dict(), output)
    label = check_equipment_completion(obj_count)
    if label:
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.putText(image, label, 
                    (image.shape[1] - label_size[0] - 40, image.shape[0] - label_size[1] - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return output, image


###################################################
# Define a route for the homepage

@app.route('/')
def index():
    return render_template('upload.html')

###################################################
# Define a route for detecting objects in an uploaded image

@app.route('/detect', methods=['POST'])
def detect_objects():
    
    image_file = request.files['image']
    image = cv2.imread(image_file)
    boxes, image_with_boxes = detect_objects_on_image(image, model)

    # Save the image with boxes to a file
    image_with_boxes = Image.fromarray(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
    filename = 'static/image_with_boxes.jpg'
    image_with_boxes.save(filename)
    
    return render_template('upload.html', objects=boxes, image_file=filename)

###################################################
# Define a route for detecting objects in a webcam stream

@app.route('/stream')
def stream():
    def generate():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            boxes, image_with_boxes = detect_objects_on_image(frame, model)
            # Encode the image as JPEG
            _, buffer = cv2.imencode('.jpg', image_with_boxes)
            frame = buffer.tobytes()
            # Use a generator to stream the encoded frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

###################################################

if __name__ == '__main__':
    app.run(debug=True)
