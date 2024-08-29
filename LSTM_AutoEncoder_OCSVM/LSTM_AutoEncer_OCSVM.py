#====================================
# Anomaly Detection
#====================================

#====================================
# Imports
#====================================
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from keras import optimizers, Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.layers import Dropout, Activation, Dense, LSTM, RepeatVector, TimeDistributed

# Set random seed for reproducibility
np.random.seed(1234)

#====================================
# Class LSTM_AutoEncoder
#====================================

class LSTM_AutoEncoder:
    
    def __init__(self,
                 sequence_length=11,
                 validation_split_pct=0.2,
                 random_seed=123,
                 batch_size=50,
                 epochs=3,
                 lstm_units=64,
                 learning_rate=1e-3,
                 data=None):
        
        self.sequence_length = sequence_length
        self.validation_split_pct = validation_split_pct
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.epochs = epochs
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate

        if data is not None:
            self.data = data
            self.load_data()
        else:
            self.prepare_data()

    def load_data(self):
        """Placeholder for loading external data."""
        pass
        
    def prepare_data(self):
        """Prepare training, validation, and test data."""
        self.X_train, self.y_train, self.X_test, self.y_test = self.split_and_prepare_data(0, 6999, 7000, 10000)
        self.X_train, self.X_valid = train_test_split(self.X_train, test_size=self.validation_split_pct, random_state=self.random_seed)
        self.timesteps = self.X_train.shape[1]
        self.num_features = self.X_train.shape[2]

    def generate_synthetic_wave(self):
        """Generate a synthetic data wave with anomalies."""
        time_points = np.arange(0.0, 100.0, 0.01)
        wave1 = np.sin(2 * 2 * np.pi * time_points) + np.random.normal(0, 0.2, len(time_points))
        wave2 = np.sin(2 * np.pi * time_points)
        wave3 = -2 * np.sin(10 * np.pi * np.arange(0.0, 5.0, 0.01))
        wave1[round(0.8 * len(time_points)):round(0.8 * len(time_points)) + 500] += wave3
        return wave1 - 2 * wave2

    def normalize(self, data):
        """Normalize the data using z-score normalization."""
        return (data - data.mean()) / data.std()

    def split_and_prepare_data(self, train_start_idx, train_end_idx, test_start_idx, test_end_idx):
        """Prepare and split the data for training and testing."""
        data = self.generate_synthetic_wave()
        sequences = np.array([data[i:i + self.sequence_length] for i in range(len(data) - self.sequence_length)])
        normalized_sequences = self.normalize(sequences)

        train_sequences = normalized_sequences[train_start_idx:train_end_idx]
        np.random.shuffle(train_sequences)
        X_train, y_train = train_sequences[:, :-1], train_sequences[:, -1]

        test_sequences = normalized_sequences[test_start_idx:test_end_idx]
        X_test, y_test = test_sequences[:, :-1], test_sequences[:, -1]

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        return X_train, y_train, X_test, y_test

    def flatten_last_timestep(self, sequences):
        """Flatten a 3D array into a 2D array using the last timestep."""
        return sequences[:, -1, :]

    def build_and_train_model(self):
        """Build, compile, and train the LSTM autoencoder model."""
        model = Sequential([
            LSTM(self.lstm_units * 4, activation='relu', input_shape=(self.timesteps, self.num_features), return_sequences=True),
            LSTM(self.lstm_units, activation='relu', return_sequences=False),
            RepeatVector(self.timesteps),
            LSTM(self.lstm_units, activation='relu', return_sequences=True),
            LSTM(self.lstm_units * 4, activation='relu', return_sequences=True),
            TimeDistributed(Dense(self.num_features))
        ])

        model.compile(optimizer=optimizers.Adam(self.learning_rate), loss='mse')
        model.summary()

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
        model_checkpoint = ModelCheckpoint(filepath="lstm_autoencoder_model.h5", save_best_only=True, verbose=0)
        tensorboard = TensorBoard(log_dir='./logs', write_graph=True, write_images=True)

        history = model.fit(
            self.X_train, self.X_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(self.X_valid, self.X_valid),
            verbose=2,
            callbacks=[early_stopping, model_checkpoint, tensorboard]
        )

        self.plot_training_history(history)
        self.model = model

    def plot_training_history(self, history):
        """Plot training and validation loss."""
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.legend()
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    def validate_model(self):
        """Validate the trained model on test data."""
        self.train_predictions = self.model.predict(self.X_train)
        self.test_predictions = self.model.predict(self.X_test)

        train_mse = np.mean(np.power(self.flatten_last_timestep(self.X_train) - self.flatten_last_timestep(self.train_predictions), 2), axis=1)
        test_mse = np.mean(np.power(self.flatten_last_timestep(self.X_test) - self.flatten_last_timestep(self.test_predictions), 2), axis=1)

        plt.plot(test_mse, 'r')
        plt.title('Test Data Mean Squared Error')
        plt.show()

    def apply_ocsvm(self, nu=0.0055, kernel="rbf", gamma=1.5):
        """Apply One-Class SVM for anomaly detection on reconstruction error."""
        train_errors = self.X_train - self.train_predictions
        flattened_train_errors = train_errors.reshape(train_errors.shape[0], -1)

        svm_classifier = svm.OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        svm_classifier.fit(flattened_train_errors)

        test_errors = self.X_test - self.test_predictions
        flattened_test_errors = test_errors.reshape(test_errors.shape[0], -1)
        y_scores = svm_classifier.predict(flattened_test_errors)

    def build_custom_lstm_model(self, layer_config):
        """Build a custom LSTM model with the given layer configuration."""
        model = Sequential([
            LSTM(layer_config['hidden_units1'], input_shape=(self.sequence_length - 1, layer_config['input_units']), return_sequences=True),
            Dropout(0.2),
            LSTM(layer_config['hidden_units2'], return_sequences=True),
            Dropout(0.2),
            LSTM(layer_config['hidden_units3'], return_sequences=False),
            Dropout(0.2),
            Dense(layer_config['output_units']),
            Activation("linear")
        ])

        model.compile(loss="mse", optimizer="rmsprop")
        return model

    def run_network(self, custom_model=None, custom_data=None):
        """Run the LSTM model on the provided data."""
        if custom_data is None:
            custom_data = self.split_and_prepare_data(0, 6999, 7000, 10000)

        if custom_model is None:
            layer_config = {'input_units': 1, 'hidden_units1': 64, 'hidden_units2': 256, 'hidden_units3': 100, 'output_units': 1}
            custom_model = self.build_custom_lstm_model(layer_config=layer_config)

        custom_model.fit(custom_data[0], custom_data[1], batch_size=self.batch_size, epochs=self.epochs, validation_split=0.05)

        train_predictions = custom_model.predict(custom_data[0])
        test_predictions = custom_model.predict(custom_data[2])

        train_mse = np.mean(np.power(custom_data[1] - train_predictions.flatten(), 2))
        test_mse = np.mean(np.power(custom_data[3] - test_predictions.flatten(), 2))

        plt.plot(test_mse, 'r')
        plt.title('Test Data Mean Squared Error')
        plt.show()

        return custom_model, custom_data[3], test_predictions.flatten(), custom_data[1], train_predictions.flatten()


#====================================
# Usage
#====================================
#auto_encoder = LSTM_AutoEncoder()
#auto_encoder.run_network()

#auto_encoder_with_params = LSTM_AutoEncoder(sequence_length=11, validation_split_pct=0.2, random_seed=123, batch_size=50, epochs=3, lstm_units=64, learning_rate=1e-3)
#auto_encoder_with_params.run_network()

