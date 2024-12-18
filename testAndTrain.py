import numpy as np
import cv2

RESIZED_IMAGE_WIDTH = 30
RESIZED_IMAGE_HEIGHT = 40


char_classifications = np.loadtxt("classifications.txt", np.float32)
flat_char_images = np.loadtxt("flattened_images.txt", np.float32)


char_classifications = char_classifications.reshape((-1, 1))


knn = cv2.ml.KNearest_create()
knn.train(flat_char_images, cv2.ml.ROW_SAMPLE, char_classifications)


img_test_sample = cv2.imread("sampleInput4.png")

if img_test_sample is None:
    print("Test image not found or cannot be opened.")
    exit()


img_gray = cv2.cvtColor(img_test_sample, cv2.COLOR_BGR2GRAY)
retval, img_thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)


img_resized = cv2.resize(img_thresh, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))


final_resized = img_resized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
final_resized = np.float32(final_resized)


retval, results, neigh_resp, dists = knn.findNearest(final_resized, k=1)


predicted_label = str(int(results[0][0]))
print(f"Predicted Label: {predicted_label}")


cv2.imshow("Test Image", img_test_sample)
cv2.waitKey(0)
cv2.destroyAllWindows()
