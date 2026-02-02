import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# 1. IMAGE PREPROCESSING & STANDARDIZATION

def load_img(path):
    """
    Standardizes raw image data by normalizing pixel values and 
    [cite_start]applying linear correction. [cite: 56, 487-507]
    """
    try:
        [cite_start]img = Image.open(path) # Open file via Pillow [cite: 57, 489]
        if img.mode == 'RGBA':
            [cite_start]img = img.convert('RGB') # Remove alpha channel if present [cite: 494-495]
        
        [cite_start]pixel_array = np.array(img) # Convert image to array [cite: 57, 499]
        [cite_start]normalized_array = pixel_array / 255.0 # Normalize values to [0, 1] [cite: 58, 501]
        
        # [cite_start]Linearization to correct for gamma compression [cite: 502-504]
        img_arr = np.where(normalized_array <= 0.04045, 
                           normalized_array / 12.92, 
                           ((normalized_array + 0.055) / 1.055) ** 2.4)
        
        [cite_start]return (img_arr * 255).astype(np.uint8) # Return as 8-bit data [cite: 506]
    except FileNotFoundError:
        [cite_start]print(f"Error: {path} not found.") # [cite: 490-491]
        return None

def rgb_to_grayscale(img_array):
    """
    [cite_start]Converts RGB data into a single-channel grayscale array. [cite: 508-524]
    """
    if img_array.ndim != 3:
        [cite_start]return img_array # Ensure image has 3 color channels [cite: 510-511]
        
    [cite_start]red, green, blue = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2] # [cite: 515-519]
    
    # [cite_start]Apply standard luminosity weights [cite: 521]
    gray_arr = (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)
    [cite_start]return gray_arr.astype(np.uint8) # [cite: 523]



# 2. LOGISTIC REGRESSION (91.84% Test Accuracy)

def calculate_sigmoid(z):
    """
    [cite_start]Calculates probability using the sigmoid function. [cite: 1316-1322]
    """
    [cite_start]si = np.clip(z, -400, 400) # Clip Z to prevent overflow [cite: 1317-1318]
    [cite_start]return 1 / (1 + np.exp(-si) + (1e-15)) # Sigmoid with epsilon to avoid division by 0 [cite: 1321]

def train_logistic_regression(X_train, y_train, learn_rate=0.4, iterations=2000):
    """
    [cite_start]Trains model via Gradient Descent to calculate loss and update weights. [cite: 1347-1362, 1374-1375]
    """
    [cite_start]m, n = X_train.shape # Get dataset dimensions [cite: 1340, 1348-1349]
    [cite_start]weight = np.random.randn(n) * 0.01 # Random initial weight [cite: 1351]
    [cite_start]bias = 0 # Initial bias [cite: 1352]
    
    for i in range(iterations):
        # [cite_start]Forward pass: Probability and loss calculation [cite: 1341, 1343]
        z = np.dot(X_train, weight) + bias
        prob = calculate_sigmoid(z)
        
        # [cite_start]Backward pass: Compute gradients [cite: 1344-1345]
        dw = (1/m) * np.dot(X_train.T, (prob - y_train))
        db = (1/m) * np.sum(prob - y_train)
        
        # [cite_start]Update parameters via gradient descent [cite: 1357-1358]
        weight -= learn_rate * dw
        bias -= learn_rate * db
        
    [cite_start]return weight, bias # [cite: 1362]



# 3. K-NEAREST NEIGHBORS (KNN)

def knn_single_prediction(new_example, X_train, y_train, k):
    """
    [cite_start]Predicts label by finding the 'k' closest neighbors in feature space. [cite: 954-971, 1058]
    """
    # [cite_start]Calculate Euclidean distance using linalg.norm [cite: 955, 960]
    distances = [np.linalg.norm(X_train[i] - new_example) for i in range(len(X_train))]
    
    # [cite_start]Sort and identify labels of the closest neighbors [cite: 961-963]
    closest_indices = np.argsort(distances)[:k]
    closest_labels = y_train[closest_indices]
    
    # [cite_start]Identify the most common label among neighbors [cite: 964-967]
    values, count = np.unique(closest_labels, return_counts=True)
    [cite_start]return values[np.argmax(count)] # [cite: 969-971]
