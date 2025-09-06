import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np



data_dict = pickle.load(open('data.pickle1', 'rb'))


lengths = [len(sample) for sample in data_dict['data']]
print("Unique feature lengths:", set(lengths))
correct_length = max(set(lengths), key=lengths.count)  # Most common length
filtered_data = []
filtered_labels = []
for sample, label in zip(data_dict['data'], data_dict['labels']):
    if len(sample) == correct_length:
        filtered_data.append(sample)
        filtered_labels.append(label)
# ----------------------------------------------------

data = np.asarray(filtered_data)
labels = np.asarray(filtered_labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier(n_estimators=200, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy*100:.2f}%")

f = open('model.p1', 'wb')
pickle.dump({'model':model}, f)
f.close()

