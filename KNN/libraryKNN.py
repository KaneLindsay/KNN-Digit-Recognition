from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main(k_neighbors, dataset):
    """
    Implementation of K-Nearest Neighbours using SKLearn library functions.

    Parameters
    ----------
    k_neighbors : int
        The number of neighbors to compare test images against
    dataset : array_like
        The dataset to work with

    """

    # Reduce 1*64 images to feature vectors.
    dataset.images = dataset.images.reshape(dataset.images.shape[0], dataset.images.shape[1] * dataset.images.shape[2])

    # Split dataset into training images and labels - (x_train, y_train) and testing images and labels (x_test, y_test)a
    x_train, x_test, y_train, y_test = train_test_split(dataset.images, dataset.target, test_size=0.25)

    classifier = KNeighborsClassifier(k_neighbors).fit(x_train, y_train)

    accuracy = round(accuracy_score(y_test, classifier.predict(x_test))*100, 2)

    print("Accuracy of "+str(k_neighbors)+" nearest neighbors is: ", str(accuracy) + "%")

    return accuracy
