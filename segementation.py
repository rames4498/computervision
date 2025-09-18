import cv2
from google.colab import files
import matplotlib.pyplot as plt

# Upload an image
uploaded = files.upload()
img_path = list(uploaded.keys())[0]

# Read image in grayscale
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detector
edges = cv2.Canny(img, 100, 200)  # (image, low_threshold, high_threshold)

# Show results
plt.subplot(1,2,1); plt.imshow(img, cmap='gray'); plt.title("Original")
plt.subplot(1,2,2); plt.imshow(edges, cmap='gray'); plt.title("Edges")
plt.show()
