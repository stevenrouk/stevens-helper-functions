import numpy as np

# scikit-image functions
from skimage import io, color, filters
from skimage.transform import resize, rotate
from skimage.color import rgb2gray

import matplotlib.pyplot as plt

def pad(arr):
    """Zero pad a 2-dimensional numpy array."""
    num_rows = arr.shape[0]
    num_cols = arr.shape[1]
    
    padded_arr = np.zeros((num_rows + 2, num_cols + 2))
    padded_arr[1:-1, 1:-1] += arr
    
    return padded_arr

def scale_normalize(image):
    """Scale values into the range of 0 to 1, where
    high absolute values are mapped to 0 or 1 and low
    absolute values are mapped in between."""
    min_val = np.min(image)
    max_val = np.max(image)
    val_range = max_val - min_val
    
    scaled = image / val_range
    if np.min(scaled) < 0:
        return scaled + np.abs(np.min(scaled))
    else:
        return scaled - np.abs(np.min(scaled))

def flip_normalize(image):
    """Divive values by the max value, then flip
    negative values to positive."""
    return np.abs(image / np.max(np.abs(image)))

def sobel_x(image, normalize='scale'):
    """Calculate the Sobel–Feldman operator x-gradient
    for edge detection. This can be plotted on its own,
    or used in conjunction with the sobel_y function
    via the sobel function."""
    x_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    padded_image = pad(image)
    processed_image = np.array([[np.sum(padded_image[j:j+3, i:i+3] * x_filter) for i in range(padded_image.shape[1]-2)] for j in range(padded_image.shape[0]-2)])
    if normalize == 'scale':
        normalized_image = scale_normalize(processed_image)
    elif normalize == 'flip':
        normalized_image = flip_normalize(processed_image)
    else:
        pass
    
    return normalized_image

def sobel_y(image, normalize='scale'):
    """Calculate the Sobel–Feldman operator y-gradient
    for edge detection. This can be plotted on its own,
    or used in conjunction with the sobel_x function
    via the sobel function."""
    y_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    padded_image = pad(image)
    processed_image = np.array([[np.sum(padded_image[j:j+3, i:i+3] * y_filter) for i in range(padded_image.shape[1]-2)] for j in range(padded_image.shape[0]-2)])
    if normalize == 'scale':
        normalized_image = scale_normalize(processed_image)
    elif normalize == 'flip':
        normalized_image = flip_normalize(processed_image)
    else:
        pass
    
    return normalized_image

def sobel(image, normalize='scale'):
    """Returns the image after applying the Sobel-Feldman
    operator. https://en.wikipedia.org/wiki/Sobel_operator"""
    gx = sobel_x(image, normalize=normalize)
    gy = sobel_y(image, normalize=normalize)
    g = np.sqrt(gx**2 + gy**2)
    
    if normalize == 'scale':
        return scale_normalize(g)
    if normalize == 'flip':
        return flip_normalize(g)

if __name__ == "__main__":
    test_jpg_filepath = 'images/sunset_image.jpg'

    # Read in .jpg file
    image = io.imread(test_jpg_filepath)

    # Display image
    io.imshow(image)
    plt.show()

    # Split out red, green, and blue channels
    image_red = image[:, :, 0]
    image_green = image[:, :, 1]
    image_blue = image[:, :, 2]

    # Convert original image to grayscale
    image_gray = rgb2gray(image)

    # Some other things we could do:
    # (see 'Image_Processing.ipynb' for more details)
    #     - plot a histogram of the pixel saturations/intensities for grayscale
    #     - plot a histogram of RGB pixel intensities
    #     - use K-Means to find color clusters, then plot just those (gives a cool
    #       comic-book / flat pastel look)
