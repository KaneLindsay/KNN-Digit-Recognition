from sklearn import datasets
import libraryKNN
import myKNN

"""
Load dataset and print dataset information. Run both libraryKNN and myKNN and compare accuracy.
"""

# Load digit dataset
dataset = datasets.load_digits()

# Description of dataset
print("DATASET INFORMATION\n-------------------")
print("Number of images: ", len(dataset.images))
print("Number of features per image: ", (len(dataset.images[1][1])*(len(dataset.images[1][2]))))
print("Attribute Information: 8x8 image of integer pixels in the range 0..16.")

# Show dimensions of data
print("Shape of image data: ", dataset.images.shape)
print("Shape of label data: ", dataset.target.shape)
print("-------------------")

k_neighbors = int(input("\nInput number of neighbors: "))

if 0 <= k_neighbors <= 100:
    print("*****************************\nSCIKIT LIBRARY IMPLEMENTATION\n*****************************")
    libraryKNN_accuracy = libraryKNN.main(k_neighbors, datasets.load_digits())
    print("*****************************\nOWN ALGORITHM IMPLEMENTATION\n*****************************")
    myKNN_accuracy = myKNN.main(k_neighbors, datasets.load_digits())
    print("-------------------\nThe difference in accuracy between KNearestNeighbour and MyKNN is",
          round(libraryKNN_accuracy - myKNN_accuracy, 2), "%")
else:
    print("Please choose a k between 1 and 100.")
