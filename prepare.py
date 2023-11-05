from typing import List, cast
from sklearn import svm, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import pandas as pd

df = pd.read_csv('/home/kazuya/ghq/github.com/acro5piano/moncargo/frontend/machine-learning/prepared.csv', delimiter=',')

X = df
y = df['Y']

splitted = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = cast(List[pd.DataFrame], splitted)

clf = MLPClassifier(hidden_layer_sizes=(i, ii))


clf.fit(X_train.iloc[:, 4:24], y_train)

predicted = clf.predict(X_test.iloc[:, 4:24])

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)
print(i, ii)

test_result = X_test.iloc[:, 0:4]
test_result['Y_predicted'] = predicted
test_result.to_csv('./result.csv')

