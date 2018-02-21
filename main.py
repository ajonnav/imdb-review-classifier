import utils
from dataset import Dataset
from data_loader import LocalDataLoader
from word2vec import SpacyWord2Vec
from classifier import SVMClassifier
from spacy.util import compounding
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, roc_auc_score
import argparse
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main(args):
    if args.mode == 'predict':
        return predict(model_path=args.model_path, model_type=args.model_type, data_path=args.data_path, limit=args.limit, clf_path=args.clf_path, clf_type=args.clf_type, save_results=args.save_results, save_dir=args.save_dir)
    elif args.mode == 'train':
        return train(model_path=args.model_path, model_type=args.model_type, data_path=args.data_path, limit=args.limit, split=args.split, clf_path=args.clf_path, clf_type=args.clf_type, epochs=args.epochs, save_results=args.save_results, save_dir=args.save_dir)
    else:
        logger.error("Invalid mode. Choose `train` or `predict`")


def train(model_path=None, model_type='spacy', data_path=None, limit=0, split=0.8, clf_path=None, clf_type='svm', epochs=10, batch_size=compounding(1., 64., 1.001), save_results=False, save_dir=''):
    logger.info("Training model")
    if not save_results:
        logger.warn("Will not save trained model and classifier. Set `--save-results` option to true to change this")
    if save_results and (save_dir == '' or save_dir is None):
        logger.error('--save-results set to true but --save-dir not specified')
        return
    train_data = load_data(data_path)
    if limit == 0:
        limit = train_data.num_samples()
    (train_text, train_cats, val_text, val_cats) = train_data.get_split_data(limit, split=split)
    w2v_model = load_model(model_path, model_type)
    clf = load_classifier(clf_path, clf_type)
    train_vectors = w2v_model.learn_embeddings(train_text, train_cats, val_text, val_cats, epochs=epochs, batch_size=batch_size)
    clf.fit(train_vectors, train_cats)
    val_vectors = w2v_model.vectorize(val_text)
    predicted_labels = clf.predict(val_vectors)
    logger.info("ROC AUC for validation set is: " + str(roc_auc_score(val_cats, predicted_labels)))
    if save_results:
        save_model(w2v_model, path=save_dir + 'model')
        save_clf(clf, path=save_dir + 'clf')
    return w2v_model, clf


def save_model(model, path=''):
    if path is None or path == '':
        logger.error("Save path for model not specified")
        raise ValueError("Save path for model not specified")
    logger.info("Saving model")
    model.save(path)


def save_clf(clf, path=''):
    if path is None or path == '':
        logger.error("Save path for classifier not specified")
        raise ValueError("Save path for classifier not specified")
    logger.info("Saving classifier")
    clf.save(path)


def predict(model_path=None, model_type='spacy', data_path=None, limit=0, clf_path=None, clf_type='svm', save_results=False, save_dir=''):
    logger.info("Generating predictions")
    if not save_results:
        logger.warn("Will not save predictions. Set `--save-results` option to true to change this")
    if save_results and (save_dir == '' or save_dir is None):
        logger.error('--save-results set to true but --save-dir not specified')
        return
    data = load_data(data_path)
    if limit == 0:
        limit = data.num_samples()
    w2v_model = load_model(model_path, model_type)
    clf = load_classifier(clf_path, clf_type)
    (text, labels) = data.get_all_data(limit=limit)
    vectors = w2v_model.vectorize(text)
    predicted_labels = clf.predict(vectors)
    if save_results:
        save_labels(predicted_labels, save_dir + 'predictions.txt')

    if len(labels) > 0:
        logger.info("ROC AUC for the given data is: " + str(roc_auc_score(labels, predicted_labels)))

    return predicted_labels


def save_labels(labels, filename):
    f = open(filename, 'w+')
    for label in labels:
        f.write(str(label) + '\n')
    f.close()


def load_model(model_path=None, model_type='spacy'):
    # Add more `if` statements here to add more models
    if model_type == 'spacy':
        logger.info("Loading spacy text categorizer")
        return SpacyWord2Vec(model_path=model_path)
    else:
        logger.error("Invalid model_type value")
        raise ValueError("Invalid model_type value")


def load_classifier(clf_path='', clf_type='svm'):
    # Add more `if` statements here to add more classifiers
    if clf_type == 'svm':
        return SVMClassifier(clf_path=clf_path)
    else:
        logger.error("Invalid clf_type value")
        raise ValueError("Invalid clf_type value")


def load_data(path=None, source='local'):
    if path is None or path == '':
        logger.error("Provide a valid data path")
        raise ValueError("Data path not specified")
    logger.info("Loading reviews from " + path)
    if source == 'local':
        data_loader = LocalDataLoader(path + 'text.txt', path + 'labels.txt')
        data = Dataset(data_loader)
        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn which reviews are good and which ones are bad')
    parser.add_argument('--mode', help='Whether to `train` a model or `predict` using a saved model', type=str, required=True)
    parser.add_argument('--model-path', help='Path to model', type=str, default=None)
    parser.add_argument('--model-type', help='Model type', type=str, default='spacy')
    parser.add_argument('--data-path', help='Directory of the data (expects <path>/text.txt and optionally <path>/labels.txt files)', type=str, default=None)
    parser.add_argument('--limit', help='Number of samples to limit training/usage to', type=int, default=0)
    parser.add_argument('--clf-path', help='Path to the classifier', type=str, default=None)
    parser.add_argument('--clf-type', help='Type of classifier', type=str, default='svm')
    parser.add_argument('--save-results', help='Option to save results', type=bool, default=False)
    parser.add_argument('--split', help='Percentage of data to use as training (vs validation). Value between 0 and 1', type=float, default=0.8)
    parser.add_argument('--save-dir', help='Directory to save results to', type=str, default='')
    parser.add_argument('--epochs', help='Number of epochs to train data for', type=int, default=10)
    args = parser.parse_args()
    main(args)
