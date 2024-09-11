import matplotlib.pyplot as plt
import numpy as np
from skimage import io, img_as_ubyte, color
from skimage.filters import threshold_multiotsu

# Read an image
image = io.imread("BSE_Image.jpg")

if len(image.shape) == 3:  # If it's a color image (3 channels)
    image = color.rgb2gray(image)

# Apply multi-Otsu threshold
thresholds = threshold_multiotsu(image, classes=5)

# Digitize (segment) original image into multiple classes.
regions = np.digitize(image, bins=thresholds)

# Normalize the segmented image to be between 0 and 255
output = img_as_ubyte(regions / np.max(regions))  # Scale to range 0-255

# Save the segmented image
plt.imsave("Otsu_Segmented.jpg", output, cmap='gray')

# Plotting and saving three separate images
# Plot the original image and save it
plt.figure()
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.savefig("Original_Image.jpg", bbox_inches='tight')

# Plot the histogram and save it
plt.figure()
plt.hist(image.ravel(), bins=255)
plt.title('Histogram')
for thresh in thresholds:
    plt.axvline(thresh, color='r')
plt.savefig("Histogram.jpg", bbox_inches='tight')

# Plot the Multi-Otsu result and save it
plt.figure()
plt.imshow(regions, cmap='Accent')
plt.title('Multi-Otsu Segmentation')
plt.axis('off')
plt.savefig("Otsu_Segmented_Result.jpg", bbox_inches='tight')

# Plotting the original image, histogram, and Multi-Otsu result
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

# Plot the original image
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original')
ax[0].axis('off')

# Plotting the histogram and the thresholds
ax[1].hist(image.ravel(), bins=255)
ax[1].set_title('Histogram')
for thresh in thresholds:
    ax[1].axvline(thresh, color='r')

# Plotting the Multi-Otsu result
ax[2].imshow(regions, cmap='Accent')  # `regions` can be shown directly here
ax[2].set_title('Multi-Otsu result')
ax[2].axis('off')

plt.subplots_adjust()
plt.show()
