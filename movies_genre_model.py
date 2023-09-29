import os
import time

from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2 import rmsprop
from tensorflow.python.keras.optimizer_v2 import adam
import movies_dataset as movies
import matplotlib.pyplot as plt

def get_kernel_dimensions(version, shape, divisor):
    image_width = shape[1]

    # original
    if version == 1:
        return 3, 3

    # square 10% width
    if version == 2:
        return int(0.1 * image_width / divisor), int(0.1 * image_width / divisor)

    # square 20% width
    if version == 3:
        return int(0.2 * image_width / divisor), int(0.2 * image_width / divisor)


def build(version, min_year, max_year, genres, ratio, epochs,
          x_train=None, y_train=None, x_validation=None, y_validation=None):
    # log
    print()
    print('version:', version)
    print('min_year:', min_year)
    print('max_year:', max_year)
    print('genres:', genres)
    print('ratio:', ratio)
    print()

    # load data if not provided
    if x_train is None or y_train is None or x_validation is None or y_validation is None:
        begin = time.time()
        x_train, y_train = movies.load_genre_data(min_year, max_year, genres, ratio, 'train')
        x_validation, y_validation = movies.load_genre_data(min_year, max_year, genres, ratio, 'validation')
        print('loaded in', (time.time() - begin) / 60, 'min.')
    else:
        print('data provided in arguments')

    print()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_validation.shape[0], 'validation samples')

    # build model
    num_classes = len(y_train[0])
    kernel_dimensions1 = get_kernel_dimensions(version, x_train.shape, 1)
    kernel_dimensions2 = get_kernel_dimensions(version, x_train.shape, 2)
    print('kernel_dimensions1:', kernel_dimensions1)
    print('kernel_dimensions2:', kernel_dimensions2)

    model = Sequential([
        Conv2D(32, kernel_dimensions1, padding='same', input_shape=x_train.shape[1:], activation='relu'),
        Conv2D(32, kernel_dimensions2, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, kernel_dimensions2, padding='same', activation='relu'),
        Conv2D(64, kernel_dimensions2, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid')
    ])

    opt = rmsprop.RMSProp(learning_rate=0.001, decay=1e-6)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())

    history = model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_data=(x_validation, y_validation))

    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    epochs = range(1,51)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if not os.path.isdir('saved_plots/'):
        os.makedirs('saved_plots/loss_plots')
        os.makedirs('saved_plots/accuracy_plots')
    loss_fig = plt.gcf()
    plt.draw()
    loss_fig.savefig('saved_plots/loss_plots/'  + str(ratio) + '_' \
                                + str(version) \
                                + '.png')

    plt.clf()
    loss_train = history.history['accuracy'] 
    loss_val = history.history['val_accuracy']
    epochs = range(1,51)
    plt.plot(epochs, loss_train, 'g', label='Training accuracy')
    plt.plot(epochs, loss_val, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy') 
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    acc_fig = plt.gcf()
    plt.draw()
    acc_fig.savefig('saved_plots/accuracy_plots/'  + str(ratio) + '_' \
                                + str(version) \
                                + '.png')

    # create dir if none
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # save model
    model_file_name = 'genres' \
                      + '_' + str(min_year) + '_' + str(max_year) \
                      + '_g' + str(len(genres)) \
                      + '_r' + str(ratio) \
                      + '_e' + str(epochs) \
                      + '_v' + str(version) + '.h5'

    model_path = os.path.join(save_dir, model_file_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
