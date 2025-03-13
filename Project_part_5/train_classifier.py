import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data_dict = pickle.load(open('Project_part_5/data.pickle', 'rb'))
data = data_dict['data']
labels = np.array(data_dict['labels'])

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Predict and calculate accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% of samples classified correctly!')

# Save the trained model
with open('Project_part_5/model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
print(" Model saved successfully!")
