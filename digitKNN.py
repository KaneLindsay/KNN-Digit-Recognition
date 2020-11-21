from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import struct
import matplotlib.pyplot as plt

# Load the digit dataset
digitDataset = datasets.load_digits()

# The pixel matrices can be reduced to 1*64 to make them easier to use.
digitDataset.images = digitDataset.images.reshape(digitDataset.images.shape[0],
                                                  digitDataset.images.shape[1] * digitDataset.images.shape[2])

# Print features and labels
print(digitDataset.images)
print(digitDataset.target)

# View dimensions of data
print(digitDataset.images.shape)
print(digitDataset.target.shape)

# Split into 75% training and 25% testing data

x_train, x_test, y_train, y_test = train_test_split(digitDataset.images, digitDataset.target, test_size=0.25)

n_neighbors = 5
classifier = KNeighborsClassifier(n_neighbors).fit(x_train, y_train)

print("Accuracy of "+str(n_neighbors)+" nearest neighbors is")
print(str(accuracy_score(y_test, classifier.predict(x_test))*100) + "%")
