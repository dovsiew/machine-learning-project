import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def accuracy(y_tes, y_pred):
    correct = 0
    for i in range(len(y_pred)):
        if(y_tes[i] == y_pred[i]):
            correct += 1
    return (correct/len(y_tes))*100

banknote_dataset = pd.read_csv("banknote-authentication_csv.csv")

print(banknote_dataset.head())
print()
features = banknote_dataset.iloc[:,:-1].values
labels = banknote_dataset.iloc[:, -1].values
print(features)
print()
print(labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

print("X_train shape: {}".format(X_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_train shape: {}".format(y_train.shape))
print("y_test shape: {}".format(y_test.shape))

scaller = StandardScaler()
X_train = scaller.fit_transform(X_train)
X_test = scaller.transform(X_test)

sklearn_knn = KNeighborsClassifier(n_neighbors=5)


sklearn_knn.fit(X_train, y_train)
pred_sklearn = sklearn_knn.predict(X_test)
print("Sklearn knn accuracy: ", accuracy(pred_sklearn, y_test))

