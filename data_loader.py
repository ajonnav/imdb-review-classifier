import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class LocalDataLoader:

    def __init__(self, text_path=None, label_path=None):
        if text_path is None or text_path == '':
            logger.error("Provide a valid text path")
            raise ValueError("Text path is empty")
        self.text_path = text_path
        self.label_path = label_path

    def load(self):
        logger.info("Loading text data")
        text_file = open(self.text_path, "r")
        text = text_file.readlines()
        text_file.close()
        if self.label_path is None or self.label_path == '':
            logger.warn("No label file specified so labels not loaded")
            return (text, [])
        else:
            try:
                logger.info("Loading label data")
                label_file = open(self.label_path, "r")
                labels = [int(label) for label in label_file.readlines()]
                label_file.close()
                return (text, labels)
            except IOError:
                logger.warn("Label file not found so labels not loaded")
                return (text, [])
