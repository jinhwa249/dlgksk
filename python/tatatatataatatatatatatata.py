import pandas as pd
df = pd.read_csv('train.csv', index_col='PassengerId')
print(df.head())

df =df[['Pclass', 'Sex', 'Age', 'SibSp', 'PArch', 'Fare', 'Survived']]
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = df.dropna()
X = df.dropna('Survived')
y = df['survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn import free, tree
model = tree.DecisionTreeClassifier()

model.fit(X_train, y_train)

y_presict = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_presict)

# from sklearn.metrics import confusion_matrix
# pd.DataFrame(
#     confusion_matrix(y_test, y_presict),
#     columns=['Predicted Not Survival', 'Predicted Survival'],
#     index=['True Not Survival', 'True Survival']
# )

