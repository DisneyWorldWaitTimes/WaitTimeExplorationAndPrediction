import importlib.util
from sklearn.linear_model import Ridge


# specify the module that needs to be
# imported relative to the path of the
# module
spec = importlib.util.spec_from_file_location("loadTrainTestPostedWaitTimes", "src/data/loadTrainTestData.py")

# creates a new module based on spec
loadTrainPosted = importlib.util.module_from_spec(spec)

# executes the module in its own namespace
# when a module is imported or reloaded.
spec.loader.exec_module(loadTrainPosted)

X_train, X_test, y_train, y_test = loadTrainPosted.loadTrainTestPostedWaitTimes()

linRidge = Ridge(alpha=20.0).fit(X_train, y_train)

print('Crime dataset')
print('ridge regression linear model intercept: {}'
     .format(linRidge.intercept_))
print('ridge regression linear model coeff:\n{}'
     .format(linRidge.coef_))
print('R-squared score (training): {:.3f}'
     .format(linRidge.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linRidge.score(X_test, y_test)))
print('Number of non-zero features: {}'
     .format(np.sum(linRidge.coef_ != 0)))
