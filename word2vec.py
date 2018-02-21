import spacy
import utils
from spacy.util import minibatch, compounding
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SpacyWord2Vec:

    def __init__(self, model_path=None):
        logger.info("Initializing spacy text categorizer")
        # If langauge path is not specified, create new language
        # Else, load language from disk
        if model_path is None or model_path == '':
            logger.info("No model path specified, creating new categorizer")
            self.nlp = spacy.load('en')
        else:
            logger.info("Model path specified, trying to load model")
            self.nlp = spacy.load(model_path)

        logger.info("Model contains following pipes: " + str(self.nlp.pipe_names))

        # If textcat is not one of the exisiting pipes, create a new textcat
        # Else, assign self.textcat to exisiting textcat
        if 'textcat' not in self.nlp.pipe_names:
            logger.info("Pipe `textcat` not found, creating new `textcat` pipe")
            textcat = self.nlp.create_pipe('textcat')
            self.nlp.add_pipe(textcat)
            # Add label
            self.nlp.get_pipe('textcat').add_label('POS')

        self.other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != 'textcat']
        self.optimizer = None
        logger.info("Finished initializing text categorizer")

    def learn_embeddings(self, train_texts, train_cats, val_texts=None, val_cats=None, force_restart=False, epochs=10, batch_size=compounding(1., 64., 1.001)):
        logger.info("Learning embeddings")
        train_texts = [utils.decode(text) for text in train_texts]
        train_cats = [{'cats': {'POS': bool(cat)}} for cat in train_cats]
        train_data = list(zip(train_texts, train_cats))
        validation = True

        if val_texts is None or val_cats is None or len(val_texts) == 0 or len(val_cats) == 0:
            logger.warn("Validation data is either not given or incomplete so evaluation of validation data set will not be done")
            validation = False

        with self.nlp.disable_pipes(*self.other_pipes):
            logger.warn('Disabled following pipes for training: ' + str(self.other_pipes))
            if self.optimizer is None or force_restart:
                self.optimizer = self.nlp.begin_training()
                logger.info("New optimizer created")
            for i in range(epochs):
                losses = {}
                batches = minibatch(train_data, size=batch_size)
                for batch in batches:
                    texts, cats = zip(*batch)
                    self.nlp.update(texts, cats, sgd=self.optimizer, drop=0.2, losses=losses)
                if validation:
                    self.validate(val_texts, val_cats)
                logger.info("Epoch " + str(i) + " finished, losses: " + str(losses))

        return self.vectorize(train_texts)

    def validate(self, val_texts, val_cats, boundary=0.5):
        val_docs = [self.nlp(utils.decode(text)) for text in val_texts]
        val_scores = [doc.cats['POS'] for doc in val_docs]
        tp = 1e-8  # True positives
        fp = 1e-8  # False positives
        fn = 1e-8  # False negatives
        tn = 1e-8  # True negatives
        for i in range(len(val_scores)):
            score = val_scores[i]
            truth = val_cats[i]
            if score >= boundary and truth == 1:
                tp += 1.
            if score < boundary and truth == 1:
                fn += 1.
            if score >= boundary and truth == 0:
                fp += 1.
            if score < boundary and truth == 0:
                tn += 1.
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        logging.info('Validation scores: Precision - {0:.3f}\tRecall - {1:.3f}'.format(precision, recall))

    def vectorize(self, texts):
        logger.info("Vectorizing given documents")
        scores = []
        with self.nlp.disable_pipes(*self.other_pipes):
            logger.warn('Disabled following pipes for vectorization: ' + str(self.other_pipes))
            if self.optimizer is not None:
                with self.nlp.get_pipe('textcat').model.use_params(self.optimizer.averages):
                    for text in texts:
                        doc = self.nlp(utils.decode(text))
                        scores.append([doc.cats['POS']])
            else:
                for text in texts:
                    doc = self.nlp(utils.decode(text))
                    scores.append([doc.cats['POS']])
        logger.info("Finished vectorizing documents")
        return scores

    def save(self, path):
        logger.info("Saving model")
        if self.optimizer is not None:
            with self.nlp.get_pipe('textcat').model.use_params(self.optimizer.averages):
                self.nlp.to_disk(path)
        else:
            self.nlp.to_disk(path)
