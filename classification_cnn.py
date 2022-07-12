
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix


MFCC_PATH = r"G:\tesi_4maggio_22\dataset_michele\log_mel_spec_10effects_1500segments.json"

MFCC_PATH_TEST = r"G:\tesi_4maggio_22\dataset_michele\log_mel_spec_TEST.json"
# X_shape: (10500, 87, 13)
# shape for mel spec: (87, 32)


def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y


def load_mapping(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    mapping = np.array(data["mapping"])
    return mapping


def plot_model_structure(model):
    keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)  # added mike
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()


def plot_history(history):

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def verify_input_shape(mfcc_file):
    shape = np.shape(mfcc_file)
    print(f"input shape is {shape}\nthe meaning is (samples, frames, coefficients)")


def prepare_dataset(test_size, validation_size):

    # load data
    X, y = load_data(MFCC_PATH)
    verify_input_shape(X)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

    # create train/validation split
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validation_size, shuffle=True)

    # add 3rd dimension
    X_train = X_train[..., np.newaxis]
    X_valid = X_valid[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def prepare_dataset_with_test_guitar(valid_size=0.2):

    print("prepare dataset with test guitar")

    # load data
    X_train, y_train = load_data(MFCC_PATH)
    X_test, y_test = load_data(MFCC_PATH_TEST)
    print("shape of X_train: ", X_train.shape)
    print("shape of X_test: ", X_test.shape)

    # create validation split
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, shuffle=True)


    # add 3rd dimension
    X_train = X_train[..., np.newaxis]
    X_valid = X_valid[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def build_model(input_shape):
    # create model
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten the output and feed into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    #model.add(keras.layers.Dense(64, activation='softmax'))

    # output layer
    model.add(keras.layers.Dense(11, activation='softmax'))
    # model.add(keras.layers.Dense(12, activation='softmax')) performa meglio così, com'è possibile??

    model.summary()
    return model

def build_model_vgg(input_shape):
    vgg_16 = keras.applications.VGG16(
        input_shape=input_shape,
        weights=None,
        classes=11
    )

    return vgg_16



def compile_model(model, learning_rate=0.001):
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def predict(model, X, y):

    # convert X (130, 13, 1) into 4D array (1, 130, 13, 1)
    X = X[np.newaxis, ...]

    # prediction = [ [0.1, 0.3, 0.2,...] ]
    y_predicted = model.predict(X)

    # extract index with max value
    predicted_index = np.argmax(y_predicted, axis=1)

    print(f"Expected index: {y}, Predicted index:{predicted_index}")


if __name__ == "__main__":

    # create train, validation and test sets
    #X_train, X_valid, X_test, y_train, y_valid, y_test = prepare_dataset(0.20, 0.15)
    X_train, X_valid, X_test, y_train, y_valid, y_test = prepare_dataset_with_test_guitar()

    # build the CNN net
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    print(f"input_shape: {input_shape}")

    model = build_model(input_shape)
    model = compile_model(model, learning_rate=0.0002)
    history = model.fit(X_train, y_train,
                        validation_data=(X_valid, y_valid),
                        batch_size=16, epochs=15, shuffle=True)

    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Accuracy on test set: {test_accuracy}")

    # plot history
    plot_history(history)

    # confusion matrix
    predictions = model.predict(x=X_test, verbose=0)
    rounded_predictions = np.argmax(predictions, axis=-1)
    cm = confusion_matrix(y_true=y_test, y_pred=rounded_predictions)

    cm_plot_labels = load_mapping(MFCC_PATH)
    print(cm_plot_labels)

    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, normalize=True, title='Confusion Matrix')

    
