from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt



def detect_objects_on_image(img, model):
    top_model = model
    results = top_model.predict(img)
    result = results[0]

    output = []
    image = img.copy()

    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([x1, y1, x2, y2, result.names[class_id], prob])

        label = f"{result.names[class_id]}: {prob}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(image, label, (x1 + 5, y1 + label_size[1] + label_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return output, image



def test():
    model = YOLO("train3/weights/best.pt")
    img = cv2.imread("static/test01.jpg")
    _, image = detect_objects_on_image(img, model)
    cv2.imshow("Image", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    test()