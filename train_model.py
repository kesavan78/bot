from sklearn.linear_model import LinearRegression
# Add code for model training here

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

