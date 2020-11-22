from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np


def main(k_neighbors, dataset):
    """
    Predict a class for every test image and find the accuracy of the algorithm using k neighbours.

    Parameters
    ----------
    k_neighbors : int
        The number of neighbors to compare test images against
    dataset : array_like
        The dataset to work with

    """

    # Show description of dataset
    print(dataset.DESCR)
    # View dimensions of data
    print(dataset.images.shape)
    print(dataset.target.shape)

    # Reduce 1*64 images to feature vectors.
    dataset.images = dataset.images.reshape(dataset.images.shape[0], dataset.images.shape[1] * dataset.images.shape[2])

    # Split dataset into training images and labels - (x_train, y_train) and testing images and labels (x_test, y_test)
    x_train, x_test, y_train, y_test = train_test_split(dataset.images, dataset.target, test_size=0.25)

    total_correct = 0
    errors = []
    i = 0
    x_pruned, y_pruned = increment_grow(k_neighbors, x_train, y_train)
    print("----------------------\nCLASSIFYING TEST DIGITS...")

    for test_image in x_test:
        prediction = predict(k_neighbors, x_pruned, y_pruned, test_image)

        if prediction == y_test[i]:
            total_correct += 1
        else:
            errors.append(
                {'Actual': y_test[i], 'Prediction': prediction})

        i += 1

    print("Correct classifications:", total_correct)
    print("Incorrect classifications:", len(x_test) - total_correct)
    print(errors)

    accuracy = (round((total_correct / i), 2) * 100)
    print(k_neighbors, "Nearest Neighbors is", accuracy, "% accurate.")


# Euclidean distance finder
def euclidean_distance(img_a, img_b):
    """
    Finds the distance between 2 images: img_a, img_b

    Parameters
    ----------
    img_a : array_like
        First feature vector to compare
    img_b : array_like
        Second feature vector to compare

    Returns
    ---------
    float
        The euclidean distance between images a and b

    """

    return sum((img_a - img_b) ** 2)


def find_best_fit(labels):
    """
    Find the total number of each label and returns the most occurring label

    Parameters
    ----------
    labels : array_like
        The set of k-closest labels

    Returns
    -------
    int
        The predicted class given using the most often occurring label

    """

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
    """
    Predict the class of a test image based on training images and labels.

    Parameters
    ----------
    k : int
        The number of neighbors to compare the test image to
    train_images : array_like
        The array of training feature vectors
    train_labels : array_like
        The array of correct labels corresponding to training feature vectors
    test_image : array_like
        The feature vector to be classified

    Returns
    -------
    int
        The predicted label for the test image

    """

    distances = []
    k_closest_labels = []

    for image in range(len(train_images)):
        distances.append(
            {'label': train_labels[image], 'distance': euclidean_distance(train_images[image], test_image)})

    # Sort the images by their euclidean distance from the test image.
    sorted_distances = sorted(distances, key=lambda k: k['distance'])

    # Append the closest images' labels to the list of closest labels
    for i in range(k):
        k_closest_labels.append(sorted_distances[i].get("label"))

    return find_best_fit(k_closest_labels)


def increment_grow(k, train_images, train_labels):
    """
    Reduce the size of the training data by using incremental growth

    Parameters
    ----------
    k : int
        The number of neighbors to compare
    train_images : array_like
        The set of feature vectors to train on
    train_labels : array_like
        The set of corresponding labels for the feature vectors

    Returns
    -------
    array_like
        The training dataset reached through incremental growth

    """

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


main(10, datasets.load_digits())
