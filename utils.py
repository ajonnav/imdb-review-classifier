from spacy.lang.en.stop_words import STOP_WORDS

TRAIN_PATH = 'data/train/'
TEST_PATH = 'data/test/'
UNKNOWN_FILE = 'data/uknown/'


def strip_stopwords(text):
    text_arr = text.split(" ")
    stripped_arr = []
    for word in text_arr:
        if word not in STOP_WORDS:
            stripped_arr.append(word)
    return ' '.join(stripped_arr)


def decode(text):
    return text.decode('utf-8')
