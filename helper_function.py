import tensorflow as tf

########################################################################

def load_and_resize_img(filename, img_shape=224, scale = True):
    '''
    Read an image from filename, turns it into a tensor 
    in the shape of (224,224,3) and rescales it,
    if scale = True.

    Parameters
    -----------
    filename(str): string filename of input image
    img_shape(int): size to resize image, default 224.
    scale(bool): whether to scale pixel values 
    '''

    # Read in the image 
    img = tf.io.read_file(filename)
    # Decode it into tensor
    img = tf.image.decode_jpeg(img)
    # Resize the img
    img = tf.image.resize(img, [img_shape, img_shape])

    # scale the image
    if scale:
        # Rescale the image (get all pixel values between 0 and 1)
        return img/255.
    else:
        return img


######################################################################

# A function to predict on images and plot them 

def pred_and_plot(model, filename, class_names):
    '''
    Imports an image located at filename, makes a prediction 
    on it with a trained model and plots the image with the predicted 
    class at the title

    Parameters
    ----------
    model : a trained tensorflow model 
    filename(str): the path of the image file
    class_names(arr) : an array of classnames

    '''

    # Import the image and preprocess it
    img = load_and_resize_img(filename)

    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Get the predicted class:
    if len(pred[0]) > 1: # check for multiclass
        pred_class = class_names[pred.argmax()] # get the index of the max value
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])] # if only one output , round

    # Plot the image and predicted class
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)

#########################################################################

    import datetime

def create_tensorboard_callback(dir_name, experiment_name):
        '''
        Creates a TensorBoard callback intance to store log files

        Stores log files with the filepath:
        'dir_name/experiment_name/current_datetime'

        Args: 
            dir_name: target directory to store TensorBoard log files
            esperiment_name: name of experiment directory( eg. resnet_model_1)
        
        '''

        log_dir = dir_name + '/' + experiment_name + '/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S%')

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                log_dir = log_dir
        )

        print(f"Saving TensorBoard log_files to {log_dir}")

        return tensorboard_callback

##############################################################################

# plot the accuracy curves and loss curves separately

import matplotlib.pyplot as plt

def plot_loss_curves(history):
    '''
    Plot the accuracy and loss curved in 2 separate plots

    Parameters:

        history : The tensorflow model history object 
    
    '''

    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    training_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']

    epochs = range(len(training_loss)) # get the epochs as list to plot along the x-axis

    plt.figure(figsize=(8,8))
    


    # plot the loss curves
    plt.subplot(2,1,1)
    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, validation_loss, label='Validation Loss')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # plot the accuracy curves
    plt.subplot(2,1,2)
    plt.plot(epochs, training_accuracy, label='Training Accuracy')
    plt.plot(epochs, validation_accuracy, label='Validation Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()


##################################################################################


def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow model History objects.
    
    Args:
      original_history: History object from original model (before new_history)
      new_history: History object from continued model training (after original_history)
      initial_epochs: Number of epochs in original_history (new_history plot starts from here) 
    """
    
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

#############################################################################

import os

def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
    dir_path (str): target directory
  
    Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


######################################################################################

# Function to evaluate: accuracy, precision, recall, f1-score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.
  Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array
  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results
