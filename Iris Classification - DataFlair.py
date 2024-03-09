import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import pickle

# Load the data
df = pd.read_csv('iris.data', names=['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels'])

# Visualize the whole dataset
sns.pairplot(df, hue='Class_labels')
plt.show()

# Seperate features and target  
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

# Calculate avarage of each features for all classes
Y_Data = np.array([np.average(X[:, i][Y==j]) for i in range(X.shape[1]) for j in np.unique(Y)])
Y_Data_reshaped = Y_Data.reshape(4, 3)
Y_Data_reshaped = np.swapaxes(Y_Data_reshaped, 0, 1)
X_axis = np.arange(len(df.columns)-1)
width = 0.25

# Plot the avarage
plt.bar(X_axis, Y_Data_reshaped[0], width, label='Setosa')
plt.bar(X_axis+width, Y_Data_reshaped[1], width, label='Versicolour')
plt.bar(X_axis+width*2, Y_Data_reshaped[2], width, label='Virginica')
plt.xticks(X_axis, df.columns[:-1])
plt.xlabel("Features")
plt.ylabel("Value in cm.")
plt.legend(bbox_to_anchor=(1.3,1))
plt.show()

# Split the data to train and test dataset.
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Support vector machine algorithm
svn = SVC(gamma='auto')
svn.fit(X_train, y_train)

# Predict from the test dataset
predictions = svn.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: {:.2f}".format(accuracy))

# A detailed classification report
print(classification_report(y_test, predictions))

# Save the model
with open('SVM.pickle', 'wb') as f:
    pickle.dump(svn, f)

# Load the model
with open('SVM.pickle', 'rb') as f:
    model = pickle.load(f)

# Predictions on new data
X_new = np.array([[3, 2, 1, 0.2], [4.9, 2.2, 3.8, 1.1], [5.3, 2.5, 4.6, 1.9]])
prediction = model.predict(X_new)
print("Prediction of Species: {}".format(prediction))