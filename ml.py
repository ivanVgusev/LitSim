import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

from progress_monitor import progress_bar
from writers_and_readers import txt_reader


"""
Choosing parameters (N for ngrams, normalized or not)
"""

# parameters
N = 3
normalized = True

if normalized:
    directory = f"values_normalised/N={N}"
else: 
    directory = f"values/N={N}"
values = []
labels = []
auth_types = ["auth1_auth1", "auth1_auth2"]
for auth_type in auth_types:
    for path, _, files in os.walk(directory):
        for file in files:
            lit_dict = eval(txt_reader(os.path.join(path, file)))
            vals = [[list(value.values())[0], list(value.values())[1], list(value.values())[2]] for value in lit_dict.values()]
            values.extend(vals)
            if auth_type == "auth1_auth1":
                labels.append(1)
            else: 
                labels.append(0)

combined = list(zip(values, labels))
shuffled = random.shuffle(combined)
values_shuffled, labels_shuffled = zip(*combined)

values_shuffled = list(values_shuffled)
labels_shuffled = list(labels_shuffled)

"""
Training the classificator
"""

count_class_0 = labels_shuffled.count(0)
count_class_1 = labels_shuffled.count(1)

X_train, X_test, y_train, y_test = train_test_split(
    values_shuffled, labels_shuffled, test_size=0.3, random_state=42, stratify=labels_shuffled
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sample_weights = compute_sample_weight(class_weight='balanced', y = y_train)

model = XGBClassifier(
    random_state=42,
    n_jobs=-1,
    objective="binary:logistic",
    eval_metric="aucpr",
    n_estimators=300,
    learning_rate=0.01,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=1.0,
    tree_method="hist"
)

print("\nОбучаем модель...")
model.fit(X_train, y_train, sample_weight=sample_weights)

y_pred = model.predict(X_test)
print("\n=== Отчёт по классификации ===")
print(classification_report(y_test, y_pred, digits=3))

accuracy = accuracy_score(y_test, y_pred)
print(f"\nТочность модели: {accuracy:.3f}")
