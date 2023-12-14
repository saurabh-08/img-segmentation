import matplotlib.pyplot as plt
import numpy as np
import cv2

# Converting RGB image to grayscale
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def plot_histogram_and_binarization(image, thresholds):
    total_plots = 1 + 1 + len(thresholds)
    plt.figure(figsize=(12, 4))

    # Plotting the grayscale image
    plt.subplot(1, total_plots, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    # Plotting histogram
    plt.subplot(1, total_plots, 2)
    plt.hist(image.ravel(), bins=256, range=[0, 256])
    plt.title('Histogram')

    # Plotting binarized img for each threshold
    for i, threshold in enumerate(thresholds):
        _, binarized = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        plt.subplot(1, total_plots, i + 3)  # Start from the 3rd position
        plt.imshow(binarized, cmap='gray')
        plt.title(f'Threshold: {threshold}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Performing Otsu thresholding
def otsu_thresholding(image):
    _, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_thresh

# Loading images
image1 = cv2.cvtColor(cv2.imread('img1.jpeg'), cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(cv2.imread('img2.jpeg'), cv2.COLOR_BGR2RGB)
image3 = cv2.cvtColor(cv2.imread('img3.jpeg'), cv2.COLOR_BGR2RGB)

gray1 = convert_to_grayscale(image1)
gray2 = convert_to_grayscale(image2)
gray3 = convert_to_grayscale(image3)

# thresholds
thresholds = [50, 100, 150]

# Plotting
plot_histogram_and_binarization(gray1, thresholds)
plot_histogram_and_binarization(gray2, thresholds)
plot_histogram_and_binarization(gray3, thresholds)

# Performing Otsu thresholding
otsu1 = otsu_thresholding(gray1)
otsu2 = otsu_thresholding(gray2)
otsu3 = otsu_thresholding(gray3)

# Displaying results
plt.figure(figsize=(12, 4))
for i, image in enumerate([otsu1, otsu2, otsu3], start=1):
    plt.subplot(1, 3, i)
    plt.imshow(image, cmap='gray')
    plt.title(f'Otsu Thresholding {i}')
    plt.axis('off')

plt.tight_layout()
plt.show()
