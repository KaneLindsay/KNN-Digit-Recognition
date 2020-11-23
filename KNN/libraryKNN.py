from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main(k_neighbors, dataset, test_data_percent):
    """
    Implementation of K-Nearest Neighbours using SKLearn library functions.

    Parameters
    ----------
    k_neighbors : int
        The number of neighbors to compare test images against
    dataset : array_like
        The dataset to work with
    test_data_percent:
        The percentage of data to use as test images

    Returns
    -------
    int
        The accuracy of classifying test images

    """

    # Reduce 1*64 images to feature vectors.
    dataset.images = dataset.images.reshape(dataset.images.shape[0], dataset.images.shape[1] * dataset.images.shape[2])

    # Split dataset into training images and labels - (x_train, y_train) and testing images and labels (x_test, y_test)a
    x_train, x_test, y_train, y_test = train_test_split(dataset.images, dataset.target, test_size=test_data_percent/100)

    classifier = KNeighborsClassifier(k_neighbors).fit(x_train, y_train)

    predictions = classifier.predict(x_test)
    total_correct = 0
    errors = []

    for i in range(len(predictions)):
        if predictions[i] == y_test[i]:
            total_correct += 1
        else:
            errors.append({'Actual': y_test[i], 'Prediction': predictions[i]})

    accuracy = round((total_correct / len(predictions)) * 100, 2)
    print("Accuracy of "+str(k_neighbors)+" nearest neighbors is: ", str(accuracy) + "%")
    print("Correct classifications:", total_correct, "\n Incorrect classifications", len(predictions)-total_correct)
    print(errors)

    return accuracy
