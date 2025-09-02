from sklearn.linear_model import LinearRegression
import numpy as np

def train_model():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 5, 7, 9, 11])  # y = 2x + 1

    model = LinearRegression()
    model.fit(X, y)
    return model

if __name__ == "__main__":
    model = train_model()
    print("Coefficient:", model.coef_[0])
    print("Intercept:", model.intercept_)
