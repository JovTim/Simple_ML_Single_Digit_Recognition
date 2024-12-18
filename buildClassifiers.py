import numpy as np
import cv2
import os

RESIZED_IMAGE_WIDTH = 30
RESIZED_IMAGE_HEIGHT = 40

flattened_images = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
int_classifications = []

for digit in range(10):
    image_folder = f"data/data/{digit}/"
    
    if not os.path.exists(image_folder):
        print(f"Folder not found: {image_folder}")
        continue

    image_count = 0

    for file_name in os.listdir(image_folder):
        if image_count >= 1100:
            break

        image_path = os.path.join(image_folder, file_name)

        img = cv2.imread(image_path)
        if img is None:
            print(f"Image not found or cannot be opened: {image_path}")
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        retval, img_thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

        img_resized = cv2.resize(img_thresh, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
        flattened_image = img_resized.flatten().astype(np.float32)
        flattened_images = np.append(flattened_images, [flattened_image], axis=0)

        int_classifications.append(ord(str(digit))) 

        image_count += 1

    print(f"Processed {image_count} images for digit {digit}")

int_classifications = np.array(int_classifications, dtype=np.float32)

np.savetxt("classifications3.txt", int_classifications)
np.savetxt("flattened_images3.txt", flattened_images)

print("Processing completed.")
