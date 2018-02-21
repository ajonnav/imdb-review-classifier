from sklearn import svm
from sklearn.externals import joblib
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SVMClassifier:

    def __init__(self, clf_path=None):
        if clf_path is None or clf_path == '':
            logger.warn("Classifier path not specified, creating new classifier")
            self.clf = svm.SVC()
        else:
            logger.info("Loading SVM classifier")
            self.load(clf_path)

    def fit(self, x, y):
        self.clf.fit(x, y)

    def predict(self, x):
        return self.clf.predict(x)

    def load(self, path):
        if path is None or path == '':
            logger.warn("Classifier path not specified, returning new classifier")
            self.clf = svm.SVC()
        self.clf = joblib.load(path)

    def save(self, path):
        if path is None or path == '':
            logger.error("Save path not specified")
        joblib.dump(self.clf, path)
