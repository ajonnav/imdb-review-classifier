import utils
import math
import random
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Dataset:

    def __init__(self, data_loader):
        if data_loader is None:
            raise ValueError('Data loader not specified')
        self.data, self.labels = data_loader.load()
        if len(self.labels) > 0:
            if len(self.data) != len(self.labels):
                print(self.data[len(self.data) - 1])
                logger.error("Found unequal number of samples and labels")
                raise ValueError("Found unequal number of samples and labels")
        logger.info("Found " + str(len(self.data)) + " samples")

    def get_split_data(self, limit=0, split=0.8):
        if limit < 0 or limit > len(self.data):
            logger.warn("`limit` value is outside the range of the number of samples, defaulting to total number of samples")
            limit = len(self.data)
        if limit == 0:
            limit = len(self.data)
        if split < 0 or split > 1:
            logger.warn("`split` is not between 0 and 1, defaulting to 0.8")
            split = 0.8
        split_index = int(math.floor(limit * split))
        return (self.data[:split_index], self.labels[:split_index], self.data[split_index:limit], self.labels[split_index:limit])

    def get_all_data(self, limit=0):
        if limit < 0 or limit > len(self.data):
            logger.warn("`limit` value is outside the range of the number of samples, defaulting to total number of samples")
            return (self.data, self.labels)
        if limit == 0:
            return (self.data, self.labels)
        else:
            return (self.data[:limit], self.labels[:limit])

    def num_samples(self):
        return len(self.data)
