import pytest
from sklearn.datasets import load_breast_cancer

def test_data_shape():
    data = load_breast_cancer(as_frame=True)
    assert data.frame.shape[0] > 0
