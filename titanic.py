import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# --- STEP 1: LOAD (Creating the data locally) ---
data = {
    'Pclass': [3, 1, 3, 1, 3, 2, 3, 1, 3, 2],
    'Age': [22.0, 38.0, 26.0, 35.0, 35.0, 54.0, 2.0, 27.0, 14.0, 4.0],
    'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'female', 'female'],
    'Survived': [0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
}
df = pd.DataFrame(data)

# --- STEP 2: CLEAN ---
df['Age'] = df['Age'].fillna(df['Age'].mean())

# --- STEP 3: CONVERT ---
df = pd.get_dummies(df, columns=['Sex'], dtype=int)

# --- STEP 4: SEPARATE AND TRAIN ---
X = df[['Pclass', 'Age', 'Sex_female', 'Sex_male']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# --- STEP 5: PREDICT ---
new_passenger = [[3, 25, 0, 1]] 
prediction = model.predict(new_passenger)

print("Did they survive? (1=Yes, 0=No):", prediction[0])
