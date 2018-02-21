# IMDB Review Classifier

This is my response to a [Scope AI](https://www.getscopeai.com/) coding challenge. The goal of the challenge is to classify IMDB movie reviews as 'positive' or 'negative'. All of the data was provided by Scope AI.

## Setup
I developed this classifier on a Mac -- my instructions may not be right for Windows/Linux machines.
To recreate my setup, after cloning this repo, I would recommend setting up a [virtualenv](https://virtualenv.pypa.io/en/stable/) and then installing all of the packages that the repo uses by running `pip install -r requirements.txt`. After installing the packages, you need to run `python -m spacy download en` to download spacy's English model. The one caveat with using the virtualenv on Mac OS is that matplotlib can act weird and throw errors ([see here](https://matplotlib.org/faq/osx_framework.html#osxframework-faq) if you do encounter errors).

## Word2Vec model

I originally thought about using a bag-of-words model -- I had used it as a part of a research project before so I was fairly comfortable with the ideas behind the model in general. However, after playing around with spacy, particularly its [TextCategorizer](https://spacy.io/api/textcategorizer) CNN, for a little bit (I hadn't used spacy before this challenge) and after thinking about the two approaches a bit more, I decided to go with the TextCategorizer.
It seems to me that a bag-of-words model or other methods that rely more heavily on count-based statistics might be good for figuring out what a document is talking about (for example, figuring out which movie a review is talking about) but may not be as good in capturing data about the tone of the document. When doing sentiment analysis, it seems like other information that would be harder to capture with a bag-of-words model would also be important. With this in mind, I decided to use the TextCategorizer model which seemed much more capable of capturing more complex relationships within the text. It also didn't hurt that the TextCategorizer CNN leverages spacy's in-built linguistic models and hooks into everything else that spacy has to offer.

## Classifer
For the classifer, I chose to use an SVM -- I'd used SVM's before and they are fairly easy to understand while still being pretty powerful. Also, as we'll see later on, the word2vec model I chose ended up converting documents into a a single score -- an SVM seemed like one of the easiest ways to separate those scores into 2 categories.

## Arriving at the results

The first thing I did was restructured the way the data was organized. I merged `positive_reviews.txt` and `negative_reviews.txt` into separate training and testing directories. I used 20,000 labeled reviews (10,000 positive and 10,000 negative reviews) as the training+validation set and set aside the other 30,000 labeled reviews for my testing set. I also created corresponding label files. While merging the reviews, I also interleaved the positive and negative reviews so that getting a validation set would be easier later.

With the training set, I split off 20% of the reviews into a validation set. With the reduced training set and the new validation set, I trained spacy's TextCategorizer model for 80 epochs. Using the scores produced by the trained categorizer for the training set, I trained an SVM classifier. Using the trained word2vec model and the trained classifier, I calculated the area under the ROC curves for the predictions for the validation sets and testing sets. Something to note here is that since the TextCategorizer produces a single score for each review (the vector is of length 1), we can also plot the ROC curves using the scores before they have passed through the SVM classifier -- this smooths out the curve.

*ROC Curve for validation data set using the categories*

![Image for ROC curve for validation data cats](https://github.com/ajonnav/imdb-review-classifier/blob/master/images/val_cat_roc.png)

*ROC Curve for validation data set using scores*

![Image for ROC curve for validation data scores](https://github.com/ajonnav/imdb-review-classifier/blob/master/images/val_score_roc.png)

*ROC Curve for testing data set using the categories*

![Image for ROC curve for testing data cats](https://github.com/ajonnav/imdb-review-classifier/blob/master/images/test_cat_roc.png)

*ROC Curve for testing data set using scores*

![Image for ROC curve for testing data scores](https://github.com/ajonnav/imdb-review-classifier/blob/master/images/test_score_roc.png)


I then generated the predictions for the unlabeled reviews.

The number 10,000 was chosen for the number of training samples because I thought that was a sufficiently large enough training data set without taking way too long to train (it still 4-5 hours on my MacBook to go through 80 epochs). The 80-20 split for the validation data was chosen because that ratio seems to be fairly common.

## Results
I've put the trained spacy model, the trained SVM classifier and the predictions for the unsupervised reviews in the `results` folder. The easiest way to load and use the model/classifier is to use the `load_model` and `load_classifier` methods in `main.py`.

The final area under the ROC curve for my testing data set was 0.78196 (going by the predicted labels).

## Structure of code

### word2vec.py
The word2vec class needs to have the methods `train_embeddings` and `vectorize`. `train_embeddings` takes training and validation texts/labels, trains the model and returns the vector embeddings for each of the training documents. In the process of training, it also logs the losses and the performance of the model against the validation set. `vectorize` takes a set of text and returns the vector embeddings for that set of text using the already trained model (you shouldn't call `vectorize` before `train_embeddings`). The class should also have `save` and `load` methods to save and load results. The class SpacyWord2Vec implements the above methods. The idea behind structuring the class like this is to allow us to easily swap different word2vec models/implementations. Adding a new model should just involve creating a new class that implements those methods and adding an entry to the `load_model` method in `main.py`.

### classifier.py

The classifier class is designed similarly to the word2vec class. Classifier classes should implement `fit`, `predict`, `save` and `load` methods (as does the SVMClassifier class) and adding a new classifier should just involve implementing those methods and adding an entry in `load_classifier` in `main.py`.

### data_loader.py

To actually load the data in, I came up with the concept of a `DataLoader`. The data loader only has a `load` method which returns The only `DataLoader` that I've implemented is a `LocalDataLoader` which loads data from your local machine but I was thinking that if your data lived in S3 or was otherwise distributed, this might be a way to abstract away the loading logic (for loading from S3, you might create an `S3DataLoader` class). The `LocalDataLoader` loads data in from a `text_path` (which contains the actual review text) and a `label_path` which contains the labels for the reviews.

### dataset.py

To actually hold the data once it is loaded, I created a `Dataset` class. In addition to holding the data, it takes care of splitting the loaded data into training and validation sets, if desired. You pass in a `DataLoader` when instantiating a `Dataset`.

### main.py

This file is the main entrypoint for the project. It assumes that you are trying to do one of two things -- train a model or predict using an old model. In addition to the train and predict methods, the file also contains a bunch of helper methods such as `load_model`, `load_data` and `load_classifier`. There are a number of flags that you can set in order to control what you want `main.py` to do. The main flag is the `--mode` flag which can be set to either `train` or `predict`. Based on what the mode is set to, the rest of the flags set different parameters:

| flag | train | predict |
| ---------------- | ----------------------------------------- | ------------------------------------------ |
| `--data-path` | Path to the training data directory; there needs to be a `text.txt` file and a `labels.txt` file | Path to the prediction data directory; there needs to be at least a `text.txt` file |
| `--model-path` | Path to model folder; only if you want to further train an existing model | Path to model to use for predictions |
| `--model-type` | Type of model, currently only `spacy` (which is the default value)| Type of model, currently only `spacy` (which is the default value) |
| `--limit` | Limits the number of samples to train on | Limits the number of samples to generate predictions for |
| `--split` | Percentage of data you want to use for training (vs validation) | No effect |
| `--clf-path` | Path to the classifier file; only if you want to further fit an existing classifier | Path to classifier to use for classification |
| `--clf-type` | Type of classifier; currently only `svm` | Type of classifier; currently only `svm` |
| `--epochs` | Number of epochs to train model for | No effect |
| `--save-results` | Whether or not to save the resulting trained model and classifier | Whether or not to save the generated predictions |
| `--save-dir` | Where to save results to (required if `--save-results` is true) | Where to save results to (required if `--save-results` is true) |
