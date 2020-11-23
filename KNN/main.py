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

# Count and print the number of entries of each label
for label in range(10):
    label_count = 0
    for entry in dataset.target:
        if label == dataset.target[entry]:
            label_count += 1
    print("Class", label, "has", label_count, "entries.")

print("Total number of entries: ", len(dataset.images))
print("Number of features per image: ", (len(dataset.images[1][1])*(len(dataset.images[1][2]))))
print("Attribute Information: 8x8 image of integer pixels in the range 0..16.")

# User inputs of value for k and the percentage of data to use as testing

k_neighbors = int(input("\nInput number of neighbors: "))
data_split_percent = int(input("\nInput percentage of testing data (1-99): "))

if 1 <= data_split_percent < 100:
    pass
else:
    data_split_percent = 25
    print("Invalid percentage - test data percentage set to 25 by default.")

if 0 < k_neighbors <= 100:
    print("\n*****************************\nSCIKIT LIBRARY IMPLEMENTATION\n*****************************\n")
    libraryKNN_accuracy = libraryKNN.main(k_neighbors, datasets.load_digits(), data_split_percent)
    print("\n*****************************\nOWN ALGORITHM IMPLEMENTATION\n*****************************\n")
    myKNN_accuracy = myKNN.main(k_neighbors, datasets.load_digits(), data_split_percent)
    print("-------------------\nThe difference in accuracy between KNearestNeighbour and MyKNN is",
          round(libraryKNN_accuracy - myKNN_accuracy, 2), "%")
else:
    print("Please choose a k between 1 and 100.")
