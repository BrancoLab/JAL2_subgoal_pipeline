from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor


def rf_model(x_train, y_train, x_test, y_test):
    """Use an esemble of decision trees to predict the output,

    n_estimators: The number of trees in the forest, i.e number of models
    random_state: Controls both the randomness of the bootstrapping of the samples
    used when building trees. Setting to zero gets rid of randomness."""

    logger.info("Fitting a random forest model")
    model = RandomForestRegressor(n_estimators=10, random_state=0)  # 100 - 500 models
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    logger.success("R2 score for the random forest model: {}".format(r2))

    return r2, y_pred

def svr_model(x_train, y_train, x_test, y_test):
    logger.info("Fitting an SVR model")
    
    # Create and fit the model
    # You can experiment with different kernels like 'linear', 'poly', 'rbf'
    model = SVR(kernel='rbf') 
    model.fit(x_train, y_train)

    # Make predictions and evaluate
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    logger.success("R2 score for the SVR model: {}".format(r2))

    return r2, y_pred

def gbr_model(x_train, y_train, x_test, y_test):
    logger.info("Fitting a Gradient Boosting Regressor model")
    
    # Create and fit the model
    # You can tune parameters like n_estimators, learning_rate, etc.
    model = GradientBoostingRegressor(n_estimators=10, learning_rate=0.01)
    model.fit(x_train, y_train)

    # Make predictions and evaluate
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    logger.success("R2 score for the Gradient Boosting model: {}".format(r2))

    return r2, y_pred


