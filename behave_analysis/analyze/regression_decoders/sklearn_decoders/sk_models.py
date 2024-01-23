from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score

# ------------------------------- Models that work -------------------------------------------------


def rf_model(x_train, y_train, x_test, y_test):
    """Use an esemble of decision trees to predict the output,

    n_estimators: The number of trees in the forest, i.e number of models
    random_state: Controls both the randomness of the bootstrapping of the samples
    used when building trees. Setting to zero gets rid of randomness."""

    logger.info("Fitting a random forest model")
    model = RandomForestRegressor(n_estimators=20, random_state=0)  # 100 - 500 models
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    logger.success("R2 score for the random forest model: {}".format(r2))

    return r2, y_pred


def gbr_model(x_train, y_train, x_test, y_test):
    logger.info("Fitting a Gradient Boosting Regressor model")

    # Create and fit the model
    # You can tune parameters like n_estimators, learning_rate, etc.
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05)
    model.fit(x_train, y_train)

    # Make predictions and evaluate
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    logger.success("R2 score for the Gradient Boosting model: {}".format(r2))

    return r2, y_pred


# ------------------------------- Tried models that did not work -----------------------------------


# Elastic net is linear so it will not work on angular data
def elastic_net_model(x_train, y_train, x_test, y_test, alpha=1.0, l1_ratio=0.5):
    """
    Fit an Elastic Net model to the data.

    Parameters:
    alpha: Constant that multiplies the penalty terms. Defaults to 1.0.
    l1_ratio: The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
              l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
              Defaults to 0.5 (equal weighting of L1 and L2 penalties).
    """

    logger.info("Fitting an Elastic Net model")

    # Create and fit the model
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(x_train, y_train)

    # Make predictions and evaluate
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    logger.success("R2 score for the Elastic Net model: {}".format(r2))

    return r2, y_pred


# SVM regression too slow
def svr_model(x_train, y_train, x_test, y_test):
    """NOTE: This model was too slow to converge, I could never
    get a result from it even with a linear kernel."""

    logger.info("Fitting an SVR model")

    # Create and fit the model
    # You can experiment with different kernels like 'linear', 'poly', 'rbf'
    model = SVR(kernel="linear")  # linear might be only one fast enough to work
    model.fit(x_train, y_train)

    # Make predictions and evaluate
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    logger.success("R2 score for the SVR model: {}".format(r2))

    return r2, y_pred
