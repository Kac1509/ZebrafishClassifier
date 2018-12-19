from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def plot_loss(history):
  # Retrieve a list of list results on training and test data
  # sets for each training epoch
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  # Get number of epochs
  epochs = range(1, len(loss)+1, 1)

  # Plot training and validation loss per epoch
  plt.plot(epochs, loss,'*-', color="b", label='train error')
  plt.plot(epochs, val_loss,'*-', color="r", label='validation error')
  plt.title('Training and validation loss')
  plt.xticks(epochs)
  plt.legend(loc=2)
  plt.grid(True)

  plt.figure()

def plot_loss_acc(history):

  # Retrieve a list of accuracy results on training and test data
  # sets for each training epoch
  acc = history.history['acc']
  val_acc = history.history['val_acc']

  # Retrieve a list of list results on training and test data
  # sets for each training epoch
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  # Get number of epochs
  epochs = range(1, len(acc)+1, 1)

  # Plot training and validation accuracy per epoch
  plt.plot(epochs, acc,'*-', color="b", label='train accuracy')
  plt.plot(epochs, val_acc,'*-', color="r", label='validation accuracy')
  plt.title('Training and validation accuracy')
  plt.xlabel('epochs')
  plt.ylabel('Accuracy')
  plt.xticks(epochs)
  plt.legend(loc=2)
  plt.grid(True)
  plt.figure()

  # Plot training and validation loss per epoch
  plt.plot(epochs, loss,'*-', color="b")
  plt.plot(epochs, val_loss,'*-', color="r")
  plt.title('Training and validation loss')
  plt.xlabel('epochs')
  plt.ylabel('Error')
  plt.xticks(epochs)
  plt.grid(True)


  
  



def cross_validation_visualization(hyperparameters, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(hyperparameters, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(hyperparameters, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("learning rate")
    plt.ylabel("loss")
    plt.title("different learning rates")
    plt.legend(loc=2)
    plt.grid(True)
    # plt.savefig("cross_validation")

    
def cross_validation_visualization1(train_sizes,train_scores_mean, train_scores_std,
                                    test_scores_mean, test_scores_std, xlabel,
                                    ylim=None):
    """visualization the curves of mse_tr and mse_te."""
    title = 'Hyperparameter Tuning using Cross-Validation'
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(xlabel)
    plt.ylabel("Error")

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="b")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="r")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="b",
             label="Training Error")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="r",
             label="Validation Error")

    plt.legend(loc="best")
    