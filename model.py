import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

train_DF = pd.read_csv("titanic.csv")

train_DF.dropna(inplace=True)

y_col = 'Survived'
X_col = ['Pclass','Age','Siblings/Spouses Aboard','Parents/Children Aboard','Fare']

y = train_DF[y_col]
X = train_DF[X_col]

model = LogisticRegression()
model.fit(X, y)

print(model.score(X, y))

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)