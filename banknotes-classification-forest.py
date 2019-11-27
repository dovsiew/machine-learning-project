import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

banknote_dataset = pd.read_csv("banknote-authentication.csv")
print(banknote_dataset.head())

samples = banknote_dataset.iloc[:, :].values
half_count = int(len(samples)/2)
np.random.shuffle(samples)
train = samples[:half_count, :-1]
train_labels = samples[:half_count, -1:]
test = samples[half_count:, :-1]
test_labels = samples[half_count:, -1:]

# Training
model = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt')
model.fit(train, np.ravel(train_labels, order='C'))

# Testing
rf_predictions = model.predict(test)

# Accuracy calculation
correct = 0
wrong = 0
for i in range(len(rf_predictions)):
    if rf_predictions[i] == np.ravel(test_labels, order='C')[i]:
        correct += 1
    else:
        wrong += 1
ratio = correct/(correct + wrong)
print(ratio)
