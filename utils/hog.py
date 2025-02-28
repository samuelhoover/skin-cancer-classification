# pyright: basic

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import exposure
from skimage.feature import hog

image = np.array(Image.open("data-bak/equalized_imgs/Malignant_3019.jpg"))
# image[image < 10] = 0

fd, hog_image = hog(
    image,
    orientations=8,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    visualize=True,
    channel_axis=-1,
    feature_vector=True,
)
print(len(fd))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis("off")
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title("Input image")

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis("off")
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title("Histogram of Oriented Gradients")
plt.show()
