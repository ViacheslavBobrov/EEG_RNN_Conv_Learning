from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, LSTM, Concatenate
from tensorflow.keras.layers import Input, TimeDistributed, Reshape, MaxPooling1D, Permute, Conv1D, BatchNormalization

from tensorflow.keras.optimizers import Adam

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import numpy as np

from utils import WeightClip, ImagePreprocessor, reformat_input, load_data, compute_interpolation_weights, \
    convert_to_images


class NeuralNetFactory:
    """
    Model creation code is repeated in some methods for the purpose of readability and easier comparison with
    the original paper. The same goes for explicit mentioning of some parameters
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def conv_net_A(self, learning_rate=0.001):
        model = Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Flatten(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(self.n_classes, activation='softmax')
        ]
        )

        adam_optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=adam_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def conv_net_B(self, learning_rate=0.001):
        model = Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Flatten(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(self.n_classes, activation='softmax')
        ]
        )

        adam_optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=adam_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def conv_net_C(self, learning_rate=0.001):
        model = Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Flatten(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(self.n_classes, activation='softmax')
        ]
        )

        adam_optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=adam_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def conv_net_D(self, learning_rate=0.001):
        model = Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Flatten(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(self.n_classes, activation='softmax')
        ]
        )

        adam_optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=adam_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def conv_net_max_pool(self, learning_rate=0.001):
        model = Sequential([
            TimeDistributed(
                Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(7, 32, 32, 3))),
            TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')),
            TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')),
            TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')),
            TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2)),
            TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')),
            TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')),
            TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2)),
            TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')),
            TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2)),
            Reshape((7, 4 * 4 * 128)),
            MaxPooling1D(pool_size=7, strides=1, data_format='channels_last'),
            Flatten(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(self.n_classes, activation='softmax')
        ]
        )

        adam_optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=adam_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def conv_net_1d_conv(self, learning_rate=0.001):
        model = Sequential([
            TimeDistributed(
                Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(7, 32, 32, 3))),
            TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')),
            TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')),
            TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')),
            TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))),
            TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')),
            TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')),
            TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))),
            TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')),
            TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))),
            TimeDistributed(Flatten()),
            Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='valid'),
            Flatten(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(self.n_classes, activation='softmax'),
        ]
        )

        adam_optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=adam_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def conv_net_lstm(self, learning_rate=0.001):
        model = Sequential([
            TimeDistributed(
                Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(7, 32, 32, 3))),
            TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')),
            TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')),
            TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')),
            TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))),
            TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')),
            TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')),
            TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))),
            TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')),
            TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))),
            TimeDistributed(Flatten()),
            LSTM(128, activation='tanh', kernel_constraint=WeightClip(100)),
            Dropout(0.5),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(self.n_classes, activation='softmax'),
        ]
        )

        adam_optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=adam_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def conv_net_lstm_1d_conv(self, learning_rate=0.001):
        input_layer = Input(shape=(7, 32, 32, 3))
        conv2d = TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))(input_layer)
        conv2d = TimeDistributed(BatchNormalization())(conv2d)
        conv2d = TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))(conv2d)
        conv2d = TimeDistributed(BatchNormalization())(conv2d)
        conv2d = TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))(conv2d)
        conv2d = TimeDistributed(BatchNormalization())(conv2d)
        conv2d = TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))(conv2d)
        maxpool2d = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(conv2d)
        maxpool2d = TimeDistributed(BatchNormalization())(maxpool2d)
        conv2d = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))(maxpool2d)
        conv2d = TimeDistributed(BatchNormalization())(conv2d)
        conv2d = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))(conv2d)
        maxpool2d = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(conv2d)
        maxpool2d = TimeDistributed(BatchNormalization())(maxpool2d)
        conv2d = TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))(maxpool2d)
        maxpool2d = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(conv2d)
        maxpool2d = TimeDistributed(BatchNormalization())(maxpool2d)
        maxpool2d = TimeDistributed(Flatten())(maxpool2d)

        lstm = LSTM(128, activation='tanh', kernel_constraint=WeightClip(100))(maxpool2d)

        conv1d = Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='valid')(maxpool2d)
        conv1d = Flatten()(conv1d)

        concat_layer = Concatenate()([lstm, conv1d])

        layer = Dropout(0.5)(concat_layer)
        layer = Dense(256, activation='relu')(layer)
        layer = BatchNormalization()(layer)
        layer = Dropout(0.5)(layer)
        output_layer = Dense(self.n_classes, activation='softmax')(layer)

        model = Model(inputs=[input_layer], outputs=[output_layer])

        adam_optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=adam_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model


def train_model(model_builder, images, labels, fold_pairs, epochs=10, batch_size=32, callbacks=[], verbose=0,
                fold_verbose=0):
    mean_val_score = 0
    mean_test_score = 0

    for i in range(len(fold_pairs)):
        model = model_builder()
        image_preprocessor = ImagePreprocessor()

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = reformat_input(images, labels, fold_pairs[i])

        X_train = image_preprocessor.fit_transform(X_train)
        X_val = image_preprocessor.transform(X_val)
        X_test = image_preprocessor.transform(X_test)

        model.fit(X_train, y_train, verbose=fold_verbose, epochs=epochs, batch_size=batch_size,
                  validation_data=(X_val, y_val),
                  callbacks=callbacks)
        y_test_predict = np.argmax(model.predict(X_test), axis=1)
        y_val_predict = np.argmax(model.predict(X_val), axis=1)

        val_accuracy_score = accuracy_score(y_val, y_val_predict)
        test_accuracy_score = accuracy_score(y_test, y_test_predict)

        mean_val_score += val_accuracy_score
        mean_test_score += test_accuracy_score

        if verbose >= 2:
            print('===========Fold {}/{}==========='.format(i + 1, len(fold_pairs)))
            print('Validation Accuracy score: ', val_accuracy_score)
            print('Test Accuracy score: ', test_accuracy_score)
        if verbose == 3:
            print('Test Precision score: ', precision_score(y_test, y_test_predict, average=None))
            print('Test Recall score: ', recall_score(y_test, y_test_predict, average=None))
            print('Test F1 score: ', f1_score(y_test, y_test_predict, average=None))

    if verbose >= 1:
        print('\nMean validation accuracy score: ', mean_val_score / len(fold_pairs))
        print('Mean test accuracy score: ', mean_test_score / len(fold_pairs))


if __name__ == '__main__':
    features, subject_numbers, fold_pairs, electrode_locations = load_data(data_folder='../Sample data')

    window_size = 192
    n_frequencies = 3
    n_gridpoints = 32
    n_windows = features.shape[1] // window_size
    channel_size = window_size // n_frequencies

    # Precompute interpolation weights because input positions (64,) and output positions (32, 32) don't change
    # For contrast, scipy.interpolation.griddata computes them during each call
    grid_x, grid_y = np.mgrid[min(electrode_locations[:, 0]):max(electrode_locations[:, 0]):n_gridpoints * 1j,
                     min(electrode_locations[:, 1]):max(electrode_locations[:, 1]):n_gridpoints * 1j]
    grid = np.zeros([n_gridpoints * n_gridpoints, 2])
    grid[:, 0] = grid_x.flatten()
    grid[:, 1] = grid_y.flatten()
    interpolation_weights = compute_interpolation_weights(electrode_locations, grid)

    # Single-Frame images generation
    average_features = np.split(features[:, :-1], n_windows, axis=1)
    average_features = np.sum(average_features, axis=0)
    average_features /= n_windows
    images = convert_to_images(average_features, n_gridpoints, n_frequencies, channel_size, interpolation_weights)
    images = np.transpose(images, (1, 2, 3, 0))
    print('images shape', images.shape)

    # Multi-Frame images generation
    images_timewin = np.array([convert_to_images(features[:, i * window_size:(i + 1) * window_size],
                                                 n_gridpoints, n_frequencies, channel_size, interpolation_weights)
                               for i in range(n_windows)])
    images_timewin = np.transpose(images_timewin, (2, 0, 3, 4, 1))
    print('images_timewin shape', images_timewin.shape)

    labels = np.squeeze(features[:, -1]) - 1
    n_classes = len(np.unique(labels))
    print('Number of classes =', n_classes)

    factory = NeuralNetFactory(n_classes=n_classes)
    train_model(factory.conv_net_A, images, labels, fold_pairs, epochs=10, verbose=2)
    train_model(factory.conv_net_B, images, labels, fold_pairs, epochs=10, verbose=2)
    train_model(factory.conv_net_C, images, labels, fold_pairs, epochs=10, verbose=2)
    train_model(factory.conv_net_D, images, labels, fold_pairs, epochs=10, verbose=2)
    train_model(factory.conv_net_max_pool, images_timewin, labels, fold_pairs, epochs=10, verbose=2)
    train_model(factory.conv_net_1d_conv, images_timewin, labels, fold_pairs, epochs=10, verbose=2)
    train_model(factory.conv_net_lstm, images_timewin, labels, fold_pairs, epochs=10, verbose=2)
    train_model(factory.conv_net_lstm_1d_conv, images_timewin, labels, fold_pairs, epochs=10, verbose=2)
