from sklearn.datasets import load_breast_cancer

def validate_data():
    data = load_breast_cancer(as_frame=True)
    print(f"Dataset loaded with shape: {data.frame.shape}")

if __name__ == "__main__":
    validate_data()
