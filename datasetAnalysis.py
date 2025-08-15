import random
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import cv2
import numpy as np
from collections import defaultdict
from skimage.measure import regionprops, label


# Set seed for reproducibility
seed = 42

# Set seeds for random number generators in NumPy and Python
np.random.seed(seed)
random.seed(seed)


# Configure plot display settings
sns.set(font_scale=1.4)
sns.set_style('white')
plt.rc('font', size=14)

# Data loading:
data = np.load("training_set.npz")

training_set = data["training_set"]
X = training_set[:, 0] # Images
y = training_set[:, 1] # Segmentation masks

X_test = data["test_set"]

print(f"Training X shape: {X.shape}")
print(f"Training y shape: {y.shape}")
print(f"Test X shape: {X_test.shape}")

category_map = {
    0: 0,  # Background
    1: 1,  # Soil
    2: 2,  # Bedrock
    3: 3,  # Sand
    4: 4,  # Big rock
}

# Calculate the correct number of classes after mapping
NUM_CLASSES = len(set(category_map.values()))


# Outliers removal (Alien)
mask_alien = y[142] # Alien image mask

# Initialize lists to store the cleaned dataset
X_cleaned = []
y_cleaned = []

for idx, mask in enumerate(y):
    if np.array_equal(mask, mask_alien):
        # If it is a duplicate mask, skip this image and mask
        continue

    # If it's not a duplicate, add the image and its corresponding mask to the cleaned dataset
    X_cleaned.append(X[idx])
    y_cleaned.append(mask)

# Convert the cleaned lists to numpy arrays
X_cleaned = np.array(X_cleaned)
y_cleaned = np.array(y_cleaned)

# Pixel normalization:
X_cleaned = X_cleaned[..., np.newaxis] / 255.0

# -------------------------------------------train-val split----------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X_cleaned,
    y_cleaned,
    test_size=0.1,
    random_state=42
)


print("Data splitted!")

print(f"\nNumber of images:")
print(f"Train: {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Test: {len(X_test)}")

# -------------------------------------------Visual Inspection----------------------------------------------
def plot_images_and_masks(X, y, start_index, end_index):
    # Ensure the range is valid
    if start_index < 0 or end_index > len(X):
        print(f"Invalid range: start_index must be >= 0 and end_index must be <= {len(X)}.")
        return

    # Number of images to plot (based on the specified range)
    num_images_to_plot = end_index - start_index

    # Set figure size for better visibility
    plt.figure(figsize=(15, num_images_to_plot * 3))  # Adjust the size of the figure

    # Plot the images with their corresponding masks in a single row
    for i in range(num_images_to_plot):
        # Plot the image
        plt.subplot(num_images_to_plot, 2, 2 * i + 1)
        plt.imshow(X[start_index + i], cmap='gray')
        plt.title(f"Image {start_index + i + 1}", fontsize=10)
        plt.axis('off')

        # Plot the mask
        plt.subplot(num_images_to_plot, 2, 2 * i + 2)
        plt.imshow(y[start_index + i], cmap='viridis', vmin=0, vmax=4)
        plt.title(f"Mask {start_index + i + 1}", fontsize=10)
        plt.axis('off')

    # Display the plot
    plt.show()

def plot_images(X, start_index, end_index):
    # Ensure the range is valid
    if start_index < 0 or end_index > len(X):
        print(f"Invalid range: start_index must be >= 0 and end_index must be <= {len(X)}.")
        return

    # Number of images to plot (based on the specified range)
    num_images_to_plot = end_index - start_index

    # Set figure size for better visibility
    plt.figure(figsize=(15, num_images_to_plot * 3))  # Adjust the size of the figure

    # Plot the images
    for i in range(num_images_to_plot):
        # Plot the image
        plt.subplot(num_images_to_plot, 2, 2 * i + 1)
        plt.imshow(X[start_index + i], cmap='gray')
        plt.title(f"Image {start_index + i + 1}", fontsize=10)
        plt.axis('off')

    # Display the plot
    plt.show()

#plot_images_and_masks(X_cleaned, y_cleaned, 100, 200)
#plot_images(X_test, 1,100)

# -------------------------------------------Analysis----------------------------------------------

# Class distribution analysis:
unique_labels, counts = np.unique(y_train, return_counts=True)
plt.bar(unique_labels, counts)
plt.xlabel("Classes")
plt.ylabel("Frequency")
plt.title("Class Distribution")
plt.savefig('../fig/original_dataset/class_distribution.png')
plt.show()

# Pixel intensity analysis:
def compute_intensity_by_label(images, masks, num_labels):
    intensity_by_label = {label: {'mean': [], 'std': []} for label in range(num_labels)}

    for img, mask in zip(images, masks):
        for label in range(num_labels):
            # Extract pixels corresponding to the current label
            pixels = img[mask == label]
            if len(pixels) > 0:
                intensity_by_label[label]['mean'].append(np.mean(pixels))
                intensity_by_label[label]['std'].append(np.std(pixels))

    return intensity_by_label

def plot_intensity_distribution_label(intensity_by_label, save_path):
    for label, stats in intensity_by_label.items():
        mean_intensity = stats['mean']
        std_intensity = stats['std']

        plt.figure(figsize=(10, 6))
        plt.hist(mean_intensity, bins=50, alpha=0.7, label='Mean Intensity')
        plt.hist(std_intensity, bins=50, alpha=0.7, label='Standard Deviation')
        plt.legend()
        plt.title(f"Pixel Intensity Distribution for Label {label}")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        plt.savefig('{}/pixelIntensity_distribution_label{}.png'.format(save_path,label))
        plt.show()

intensity_by_label = compute_intensity_by_label(X_train, y_train, NUM_CLASSES)
plot_intensity_distribution_label(intensity_by_label, save_path='../fig/original_dataset')

def compute_intensity(images):
    intensity = {'mean': [], 'std': []}
    for img in images:
        if img.size > 0:  # Ensure the image is not empty
            intensity['mean'].append(np.mean(img))
            intensity['std'].append(np.std(img))

    return intensity


# Function to plot intensity distributions
def plot_intensity_distribution(intensity):
    mean_intensity = intensity['mean']
    std_intensity = intensity['std']

    # Create subplots for clarity
    plt.figure(figsize=(12, 6))

    # Mean intensity histogram
    plt.subplot(1, 2, 1)
    plt.hist(mean_intensity, bins=50, alpha=0.7, color='blue')
    plt.title("Mean Intensity Distribution")
    plt.xlabel("Mean Intensity")
    plt.ylabel("Frequency")

    # Standard deviation histogram
    plt.subplot(1, 2, 2)
    plt.hist(std_intensity, bins=50, alpha=0.7, color='orange')
    plt.title("Standard Deviation Intensity Distribution")
    plt.xlabel("Standard Deviation")
    plt.ylabel("Frequency")

    # Display the plots
    plt.tight_layout()
    plt.show()

intensity = compute_intensity(X_train)
plot_intensity_distribution(intensity)


sizes = []
for mask in y_train:
    labeled_mask = label(mask)
    props = regionprops(labeled_mask)
    sizes.extend([prop.area for prop in props])

plt.hist(sizes, bins=50)
plt.title("Object Size Distribution in Masks")
plt.savefig('../fig/original_dataset/obj_size_distribution.png')
plt.show()


# Class heatmaps (normalized between 0 and 0.5):
def create_class_heatmap(masks, class_label):
    heatmap = np.zeros_like(masks[0], dtype=float)
    for mask in masks:
        heatmap += (mask == class_label).astype(float)
    return heatmap / len(masks)

for class_label in np.unique(y_train):
    heatmap = create_class_heatmap(y_train, class_label)
    plt.imshow(heatmap, cmap='hot', vmin=0, vmax=0.5)  # Fixed color range
    plt.title("Spatial Heatmap for Class {} (normalized)".format(class_label))
    plt.colorbar()
    plt.savefig('../fig/original_dataset/class_heatmap_norm{}.png'.format(class_label))
    plt.show()

# Class heatmaps (adaptive range):
def create_class_heatmap(masks, class_label):
    heatmap = np.zeros_like(masks[0], dtype=float)
    for mask in masks:
        heatmap += (mask == class_label).astype(float)
    return heatmap / len(masks)

for class_label in np.unique(y_train):
    heatmap = create_class_heatmap(y_train, class_label)
    plt.imshow(heatmap, cmap='hot')  # Adaptive color range
    plt.title("Spatial Heatmap for Class {} (adaptive range)".format(class_label))
    plt.colorbar()
    plt.savefig('../fig/original_dataset/class_heatmap_adaptive{}.png'.format(class_label))
    plt.show()


correlations = []
for label in unique_labels:
    # Create binary mask for the current label
    mask = (y_train == label).flatten()
    pixel_values = X_train.flatten()

    assert len(mask) == len(pixel_values), "Lengths of mask and pixel values must match."

    # Compute Pearson correlation
    correlation, _ = pearsonr(pixel_values, mask.astype(float))
    correlations.append(correlation)

plt.bar(unique_labels, correlations)
plt.xlabel("Classes")
plt.ylabel("Correlation with Intensity")
plt.savefig('../fig/original_dataset/correlation_analysis.png')
plt.show()

# Co-occurrence analysis:
def co_occurrence(masks, num_classes=NUM_CLASSES):
    co_matrix = np.zeros((num_classes, num_classes), dtype=int)  # Initialize co-occurrence matrix
    for mask in masks:
        unique_labels = np.unique(mask).astype(int)
        for i in unique_labels:
            for j in unique_labels:
                if i != j:  # Ignore diagonal
                    co_matrix[i, j] += 1
    return co_matrix

co_matrix = co_occurrence(y_train)

plt.figure(figsize=(8, 6))
sns.heatmap(
    co_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[0,1,2,3,4],
    yticklabels=[0,1,2,3,4]
)
plt.title("Class Co-Occurrence Heatmap")
plt.xlabel("Class")
plt.ylabel("Class")
plt.savefig('../fig/original_dataset/co_occurrence_analysis.png')
plt.show()


# Function to compute the edge density of parts of an image associated to a certain label
def compute_edge_density(img, binary_mask, threshold1, threshold2):
    scaled_img = (img * 255).astype(np.uint8)
    edges = cv2.Canny(scaled_img, threshold1, threshold2)
    masked_edges = edges * binary_mask
    num_edge_pixels = np.sum(masked_edges > 0)
    num_total_pixels = np.sum(binary_mask > 0)
    return num_edge_pixels / num_total_pixels if num_total_pixels > 0 else 0

# Dictionary to store the edge densities for each class
thresholds = [(20, 60), (30, 70), (40, 80), (50, 90), (60, 100)]
class_edge_densities = defaultdict(lambda: {t: [] for t in thresholds})

# Loop through each image and its corresponding segmentation mask
for img, mask in zip(X_train, y_train):
    for class_label in range(5):
        binary_mask = (mask == class_label).astype(np.uint8)

        # Compute edge densities for multiple thresholds
        for threshold1, threshold2 in thresholds:
            if (threshold1, threshold2) not in class_edge_densities[class_label]:
                class_edge_densities[class_label][threshold1, threshold2] = []

            edge_density = compute_edge_density(img, binary_mask, threshold1, threshold2)
            class_edge_densities[class_label][threshold1, threshold2].append(edge_density)

# Calculate mean edge density for each class across all thresholds
mean_edge_densities = {}
for class_label in range(5):
    for threshold1, threshold2 in class_edge_densities[class_label].keys():
        mean_edge_densities[(class_label, threshold1, threshold2)] = np.mean(class_edge_densities[class_label][threshold1, threshold2])

threshold_labels = [f'Thresholds {t1}-{t2}' for t1, t2 in thresholds]
x_labels = list(range(5))  # Class labels: 0, 1, 2, 3, 4

plt.figure(figsize=(10, 6))
for threshold, label in zip(thresholds, threshold_labels):
    means = [mean_edge_densities[(class_label, threshold[0], threshold[1])] for class_label in range(5)]
    plt.scatter(x_labels, means, label=label, s=100)  # Use bigger dots

plt.xticks(x_labels)
plt.xlabel('Class Label')
plt.ylabel('Mean Edge Density')
plt.title('Mean Edge Density per Class and Threshold Pair')
plt.legend()
plt.savefig('../fig/original_dataset/meanEdgeDensity.png')
plt.show()

