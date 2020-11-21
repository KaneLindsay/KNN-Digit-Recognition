from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# Load the digit dataset
digitDataset = datasets.load_digits()

print(digitDataset.DESCR)

# The pixel matrices must be reduced to a 64 feature vector to use them.
digitDataset.images = digitDataset.images.reshape(digitDataset.images.shape[0],
                                                  digitDataset.images.shape[1] * digitDataset.images.shape[2])

# Split dataset into training images and labels - (x_train, y_train) and testing images and labels (x_test, y_test)
x_train, x_test, y_train, y_test = train_test_split(digitDataset.images, digitDataset.target, test_size=0.25)


# Euclidean distance finder
def euclidean_distance(img_a, img_b):
    # Finds the distance between 2 images: img_a, img_b
    return sum((img_a - img_b) ** 2)


def find_best_fit(labels):
    counts = np.zeros(10, dtype=int)
    best = 0

    # Count the labels
    for i in range(len(labels)):
        counts[labels[i]] += 1

    # Find the most often occurring label from the counts
    for i in range(len(counts)):
        if counts[i] > best:
            best = i

    return best


def predict(k, train_images, train_labels, test_image):
    # Create arrays for all point's euclidean distance from the test image and the closest points.
    distances = []
    k_closest_labels = []

    for image in range(len(train_images)):
        distances.append({'label': train_labels[image], 'distance': euclidean_distance(train_images[image], test_image)})

    # Sort the images by their euclidean distance from the test image.
    sorted_distances = sorted(distances, key=lambda k: k['distance'])

    # Append the closest images' labels to the list of closest labels
    for i in range(k):
        k_closest_labels.append(sorted_distances[i].get("label"))

    return find_best_fit(k_closest_labels)


def increment_grow(k, train_images, train_labels):
    print("----------------------\nINCREMENTALLY GROWING LABELS...")
    # Edited instance-based learning: Incremental Growth
    pruned_images = [train_images[0]]
    pruned_labels = [train_labels[0]]
    possible_neighbors: int

    for i in range(len(train_labels)):

        if len(pruned_labels) < k:
            possible_neighbors = len(pruned_labels)
        else:
            possible_neighbors = k

        prune_result = predict(possible_neighbors, pruned_images, pruned_labels, train_images[i])

        if prune_result != train_labels[i]:
            pruned_images.append(train_images[i])
            pruned_labels.append(train_labels[i])

    print("Training elements before pruning: ", len(train_labels))
    print("Training elements after pruning: ", len(pruned_labels))

    return pruned_images, pruned_labels


k_neighbors = 7
totalCorrect = 0
i = 0
x_pruned, y_pruned = increment_grow(k_neighbors, x_train, y_train)
print("----------------------\nCLASSIFYING TEST DIGITS...")

for test_image in x_test:
    prediction = predict(k_neighbors, x_pruned, y_pruned, test_image)

    if prediction == y_test[i]:
        totalCorrect += 1

    i += 1

print("Correct classifications:", totalCorrect)
print("Incorrect classifications:", len(x_test)-totalCorrect)
accuracy = (round((totalCorrect / i), 2) * 100)

print(k_neighbors, "Nearest Neighbors is", accuracy, "% accurate.")
