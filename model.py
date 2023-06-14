import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import export_graphviz
from sklearn.preprocessing import StandardScaler
from dataset import final_df

X = final_df.drop(columns=['position'])
y = final_df['position']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Accuracy: ', accuracy_score(y_test, y_pred))
print('RMSE: ', mean_squared_error(y_test, y_pred))

# Guardamos el modelo

pickle.dump(model, open('../ml/model.sav', 'wb'))

# Visualizamos el modelo

estimator = model.estimators_[5]

export_graphviz(estimator, out_file='../ml/tree.dot',
                feature_names=X.columns,
                rounded=True, proportion=False,
                precision=2, filled=True)

## dot -Tpng tree.dot -o tree.png -Gdpi=600
