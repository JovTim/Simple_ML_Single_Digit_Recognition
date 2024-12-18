import numpy as np
import cv2
import os

# Constants for resized image dimensions
RESIZED_IMAGE_WIDTH = 30
RESIZED_IMAGE_HEIGHT = 40

# Arrays to store processed image data and corresponding classifications
flattened_images = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
int_classifications = []

# Loop through digits 0-9 to process their corresponding image folders
for digit in range(10):
    image_folder = f"data/data/{digit}/"
    
    if not os.path.exists(image_folder):
        print(f"Folder not found: {image_folder}")
        continue

    image_count = 0

    for file_name in os.listdir(image_folder):
        if image_count >= 1100:
            break
        # Construct the full path to the image
        image_path = os.path.join(image_folder, file_name)

        img = cv2.imread(image_path)
        if img is None:
            print(f"Image not found or cannot be opened: {image_path}")
            continue

        # Convert the image to grayscale for easier processing and analysis
        # Grayscale reduces image complexity by representing pixel intensity as a single channel (0-255), simplifying thresholding and resizing.
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply binary thresholding to create a black-and-white image
        retval, img_thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

        # Resize the thresholded image to the standard dimensions
        img_resized = cv2.resize(img_thresh, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

        # Flatten the resized image into a 1D array and append to the dataset
        flattened_image = img_resized.flatten().astype(np.float32)
        flattened_images = np.append(flattened_images, [flattened_image], axis=0)

        # Append the ASCII value of the digit to the classifications list
        int_classifications.append(ord(str(digit))) 

        image_count += 1

    print(f"Processed {image_count} images for digit {digit}")

int_classifications = np.array(int_classifications, dtype=np.float32)

np.savetxt("classifications3.txt", int_classifications)
np.savetxt("flattened_images3.txt", flattened_images)

print("Processing completed.")
