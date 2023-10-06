import cv2
import os

# Path to the folder containing the images
folder_path = "burning_forest"

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # Load the image using OpenCV
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        # Resize the image to 640x640
        resized_img = cv2.resize(img, (640, 640))

        # Save the resized image
        resized_img_path = os.path.join(folder_path, f"resized_{filename}")
        cv2.imwrite(resized_img_path, resized_img)