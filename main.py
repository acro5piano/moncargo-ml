from typing import List, cast

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

df = pd.read_csv(
    "/home/kazuya/ghq/github.com/acro5piano/moncargo/frontend/machine-learning/prepared.csv",
    delimiter=",",
)

X = df
y = df["Y"]

splitted = train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = cast(List[pd.DataFrame], splitted)

clf = MLPClassifier(
    hidden_layer_sizes=(100, 100),
    learning_rate="adaptive",
    solver="lbfgs",
    max_iter=400,
)
clf.fit(X_train.iloc[:, 4:24], y_train)

predicted = clf.predict(X_test.iloc[:, 4:24])

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)
# print(i, ii)

test_result = X_test.iloc[:, 0:4]
test_result["Y_predicted"] = predicted
test_result.to_csv("./result.csv", index=False)
