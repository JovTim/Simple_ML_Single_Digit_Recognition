import numpy as np
import cv2

RESIZED_IMAGE_WIDTH = 30
RESIZED_IMAGE_HEIGHT = 40


char_classifications = np.loadtxt("classifications.txt", np.float32)
flat_char_images = np.loadtxt("flattened_images.txt", np.float32)

# Reshape classifications to a single column for training compatibility
char_classifications = char_classifications.reshape((-1, 1))

# Initialize and train the k-Nearest Neighbors (k-NN) classifier
knn = cv2.ml.KNearest_create()
knn.train(flat_char_images, cv2.ml.ROW_SAMPLE, char_classifications)


img_test_sample = cv2.imread("sampleInput4.png")

if img_test_sample is None:
    print("Test image not found or cannot be opened.")
    exit()

# Convert the test image to grayscale for consistent preprocessing
img_gray = cv2.cvtColor(img_test_sample, cv2.COLOR_BGR2GRAY)
# Apply binary thresholding to the grayscale test image
retval, img_thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

# Resize the thresholded test image to match the training image dimensions
img_resized = cv2.resize(img_thresh, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

# Flatten the resized test image into a 1D array for prediction
final_resized = img_resized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
final_resized = np.float32(final_resized)

# Use the k-NN classifier to predict the label of the test image
retval, results, neigh_resp, dists = knn.findNearest(final_resized, k=1)

# Extract and display the predicted label
predicted_label = str(int(results[0][0]))
print(f"Predicted Label: {predicted_label}")


cv2.imshow("Test Image", img_test_sample)
cv2.waitKey(0)
cv2.destroyAllWindows()
