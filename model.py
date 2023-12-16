import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
import ast

# Load data into a DataFrame
data = pd.read_csv("data.csv")

data['Position'] = data['Position'].apply(ast.literal_eval)
data['Velocity'] = data['Velocity'].apply(ast.literal_eval)

data[['PosX', 'PosY', 'PosZ']] = pd.DataFrame(data['Position'].tolist(), index=data.index)
data[['VelX', 'VelY', 'VelZ']] = pd.DataFrame(data['Velocity'].tolist(), index=data.index)

data = pd.get_dummies(data, columns=['State'])

X = data.drop(columns=['AgentID', 'Position', 'Velocity', 'Density'])
y = data['Density'] > 3  # Binary classification: Density > 2 or not

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ast
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier
#from catboost import CatBoostClassifier

# Dictionary of models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
#    'LightGBM': LGBMClassifier(random_state=42),
#    'CatBoost': CatBoostClassifier(random_state=42, verbose=0),  }
}

results = {}

for name, model in models.items():
   # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

   # Train model
    model.fit(X_train, y_train)

   # Evaluate model
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True)
    results[name] = report['weighted avg']['f1-score']  # Storing F1-score
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

print(sorted_results)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

X = data.drop(columns=['AgentID', 'Position', 'Velocity', 'Density'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

corr_matrix = X_scaled_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()
