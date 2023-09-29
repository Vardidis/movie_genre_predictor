import os
import time

import tensorflow.keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt

import movies_dataset as movies


def get_kernel_dimensions(version, shape, divisor):
	image_width = shape[1]

	# original
	if version == 1: return 6, 6

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
		print('no data provided in arguments')

	print()
	print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
	print('Test: X=%s, y=%s' % (x_validation.shape, y_validation.shape))	
	print(x_validation.shape[0], 'validation samples')

	# build model
	num_classes = len(y_train[0])
	kernel_dimensions1 = get_kernel_dimensions(version, x_train.shape, 1)
	kernel_dimensions2 = get_kernel_dimensions(version, x_train.shape, 2)
	print('kernel_dimensions1:', kernel_dimensions1)
	print('kernel_dimensions2:', kernel_dimensions2)
	#new 
	print('x_train.shape[1:]=', x_train.shape[1:])

	model = Sequential([
		Conv2D(32, kernel_dimensions1, padding='same', input_shape=x_train.shape[1:], activation='relu'),
		Conv2D(32, kernel_dimensions1, activation='relu'),
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

	'''
    compile the model using binary cross-entropy rather than categorical cross-entropy.
    This may seem counterintuitive for multi-label classification; however, the goal is to 
    treat each output label as an independent Bernoulli distribution and we want to penalize 
    each output node independently.	
    '''
	opt = tensorflow.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	print(model.summary())

	history = model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_data=(x_validation, y_validation))

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

	loss_train = history.history['loss']
	loss_val = history.history['val_loss']
	epochs = range(0,60)
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
	epochs = range(0,60)
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

	model_path = os.path.join(save_dir, model_file_name)
	model.save(model_path)
	print('Saved trained model at %s ' % model_path)
