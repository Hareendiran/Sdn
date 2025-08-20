from ml_model import train_model

def test_model_training():
    model = train_model()
    assert abs(model.coef_[0] - 2) < 0.1
    assert abs(model.intercept_ - 1) < 0.1
