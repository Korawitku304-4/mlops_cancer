import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_and_predict():
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    target_names = data.target_names

    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    sample_0 = df[df['target'] == 0].iloc[0]
    sample_1 = df[df['target'] == 1].iloc[0]
    samples = pd.DataFrame([sample_0, sample_1])

    X_samples = samples.drop('target', axis=1)
    y_true = samples['target'].values

    predictions = model.predict(X_samples)

    for i in range(len(predictions)):
        pred_label = target_names[int(predictions[i])]
        true_label = target_names[int(y_true[i])]
        is_correct = "Correct" if predictions[i] == y_true[i] else "Incorrect"
        print(f"True: {true_label} ^| Predicted: {pred_label} ^| Result: {is_correct}")

if __name__ == "__main__":
    load_and_predict()
