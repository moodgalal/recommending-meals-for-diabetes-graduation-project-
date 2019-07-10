###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from time import time

sns.set()

def plot_histogram(df, features, preprocessed=False):
    fig = plt.figure(figsize=(20,10))
    for i, feature in enumerate(features, 1):
        ax = fig.add_subplot(2, 3, i)
        if(preprocessed):
            ax.hist(df[feature], bins=30, range=(0, 1))
        else:
            ax.hist(df[feature], bins=50, range=(0, 1000))

        ax.set_title('%s feature histogram' %feature)

    fig.suptitle("Histogram viasuals", fontsize = 16, y = 1.03)
    fig.tight_layout()

    
def plot_box_plot(df, features):
    fig = plt.figure(figsize=(20,10))
    for i, feature in enumerate(features, 1):
        ax = fig.add_subplot(2, 3, i)
        sns.boxplot(data=df[feature])
        ax.set_title('%s feature box plot' %feature)

    fig.suptitle("Box plot viasuals", fontsize = 16, y = 1.03)
    fig.tight_layout()
    
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
