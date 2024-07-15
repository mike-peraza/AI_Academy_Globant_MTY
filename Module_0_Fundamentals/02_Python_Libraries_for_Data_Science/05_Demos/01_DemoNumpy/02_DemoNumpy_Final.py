############################################################################################
# Python code example for reading the bits of an Image of 600 x 499 and store it using Numpy
# arrays. Then doing some transformation to apply a grayscale operation and save it to a 
# new image.
############################################################################################

# Step 1. We first need to import the libraries this demo is going to use
import numpy as np
# Install first: pip install pillow
from PIL import Image

# Step 2. Load the image using NumPy
image = np.array(Image.open('Module_0_Fundamentals/02_Python_Libraries_for_Data_Science/05_Demos/Husky.jpg'))

# Display the shape of the array
print("Shape of the image array:", image.shape)

# Display the image array
print("Original Image array:")
print(image)

# Step 3. Convert the image to grayscale
grayscale_image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

# Display the image array
print("Grayscale Image array:")
print(grayscale_image)

# Step 4. Perform some operations for the new grayscale image.
# Convert the grayscale image to a PIL Image
grayscale_image_pil = Image.fromarray(grayscale_image.astype(np.uint8))

# Resize the grayscale image to 500x500 pixels
resized_grayscale_image = grayscale_image_pil.resize((600, 400))
# rotated_grayscale_image = resized_grayscale_image.rotate(90)

# Step 5. Save the new image file.
# Save the grayscale image to a new file
resized_grayscale_image.save('Module_0_Fundamentals/02_Python_Libraries_for_Data_Science/05_Demos/Husky_Grayscale.jpg')

# Display the size of the resized grayscale image
print("Size of the resized grayscale image:", resized_grayscale_image.size)
