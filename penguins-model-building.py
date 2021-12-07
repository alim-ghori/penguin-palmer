import pickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
penguins = pd.read_csv('penguins_cleaned.csv')

# Ordinal Feature Encoding
df = penguins.copy()
target = 'species'
encode = ['sex', 'island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}


def target_encode(val):
    return target_mapper[val]


df['species'] = df['species'].apply(target_encode)

# Separating X and Y
X = df.drop('species', axis=1)
Y = df['species']

# Building Model
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving Model
pickle.dump(clf, open('penguins-clf-model.pkl', 'wb'))
