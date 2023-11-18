from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load the RGB image
image_path = "leaves.jpg"
rgb_image = Image.open(image_path)

# Convert the RGB image to a NumPy array for manipulation
rgb_array = np.array(rgb_image)

# Lightness Method
lightness_image = np.clip((np.max(rgb_array, axis=2) + np.min(rgb_array, axis=2)) / 2, 0, 255).astype(np.uint8)

# Average Method
average_image = np.mean(rgb_array, axis=2).astype(np.uint8)

# Luminosity Method
luminosity_image = np.dot(rgb_array[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

# Plotting the three variants side by side
fig, axs = plt.subplots(1, 4, figsize=(15, 5))

# Original RGB image
axs[0].imshow(rgb_image)
axs[0].set_title("Original RGB")

# Lightness Method
axs[1].imshow(lightness_image, cmap="gray")
axs[1].set_title("Lightness")

# Average Method
axs[2].imshow(average_image, cmap="gray")
axs[2].set_title("Average")

# Luminosity Method
axs[3].imshow(luminosity_image, cmap="gray")
axs[3].set_title("Luminosity")

# Remove axis labels
for ax in axs:
    ax.axis('off')

plt.show()
