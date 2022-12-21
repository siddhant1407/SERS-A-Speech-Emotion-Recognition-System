import os
import numpy as np

from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score, mean_absolute_error

from emotion_recognition import EmotionRecognizer
from utils import get_first_letters, AVAILABLE_EMOTIONS, extract_feature, get_dropout_str


class DeepEmotionRecognizer(EmotionRecognizer):

    def __init__(self, **kwargs):
        
        # init EmotionRecognizer
        super().__init__(**kwargs)

        self.n_rnn_layers = kwargs.get("n_rnn_layers", 2)
        self.n_dense_layers = kwargs.get("n_dense_layers", 2)
        self.rnn_units = kwargs.get("rnn_units", 128)
        self.dense_units = kwargs.get("dense_units", 128)
        self.cell = kwargs.get("cell", LSTM)


        self.dropout = kwargs.get("dropout", 0.3)
        self.dropout = self.dropout if isinstance(self.dropout, list) else [self.dropout] * ( self.n_rnn_layers + self.n_dense_layers )
        # number of classes ( emotions )
        self.output_dim = len(self.emotions)

        # optimization attributes
        self.optimizer = kwargs.get("optimizer", "adam")
        self.loss = kwargs.get("loss", "categorical_crossentropy")

        # training attributes 
        self.batch_size = kwargs.get("batch_size", 64)
        self.epochs = kwargs.get("epochs", 500)
        
        # the name of the model
        self.model_name = ""
        self._update_model_name()

        # init the model
        self.model = None

        # compute the input length
        self._compute_input_length()

        # boolean attributes
        self.model_created = False

    def _update_model_name(self):

        # get first letters of emotions, for instance:
        emotions_str = get_first_letters(self.emotions)
        # 'c' for classification & 'r' for regression
        problem_type = 'c' if self.classification else 'r'
        dropout_str = get_dropout_str(self.dropout, n_layers=self.n_dense_layers + self.n_rnn_layers)
        self.model_name = f"{emotions_str}-{problem_type}-{self.cell.__name__}-layers-{self.n_rnn_layers}-{self.n_dense_layers}-units-{self.rnn_units}-{self.dense_units}-dropout-{dropout_str}.h5"

    def _get_model_filename(self):
        return f"results/{self.model_name}"

    def _model_exists(self):
        
        filename = self._get_model_filename()
        return filename if os.path.isfile(filename) else None

    def _compute_input_length(self):

        if not self.data_loaded:
            self.load_data()
        self.input_length = self.X_train[0].shape[1]

    def _verify_emotions(self):
        super()._verify_emotions()
        self.int2emotions = {i: e for i, e in enumerate(self.emotions)}
        self.emotions2int = {v: k for k, v in self.int2emotions.items()}

    def create_model(self):
        """
        Constructs the neural network based on parameters passed.
        """
        if self.model_created:
            # model already created, why call twice
            return

        if not self.data_loaded:
            # if data isn't loaded yet, load it
            self.load_data()
        
        model = Sequential()

        # rnn layers
        for i in range(self.n_rnn_layers):
            if i == 0:
                # first layer
                model.add(self.cell(self.rnn_units, return_sequences=True, input_shape=(None, self.input_length)))
                model.add(Dropout(self.dropout[i]))
            else:
                # middle layers
                model.add(self.cell(self.rnn_units, return_sequences=True))
                model.add(Dropout(self.dropout[i]))

        if self.n_rnn_layers == 0:
            i = 0

        # dense layers
        for j in range(self.n_dense_layers):
            # if n_rnn_layers = 0, only dense
            if self.n_rnn_layers == 0 and j == 0:
                model.add(Dense(self.dense_units, input_shape=(None, self.input_length)))
                model.add(Dropout(self.dropout[i+j]))
            else:
                model.add(Dense(self.dense_units))
                model.add(Dropout(self.dropout[i+j]))
                
        if self.classification:
            model.add(Dense(self.output_dim, activation="softmax"))
            model.compile(loss=self.loss, metrics=["accuracy"], optimizer=self.optimizer)
        else:
            model.add(Dense(1, activation="linear"))
            model.compile(loss="mean_squared_error", metrics=["mean_absolute_error"], optimizer=self.optimizer)
        
        self.model = model
        self.model_created = True
        if self.verbose > 0:
            print("[+] Model created")

    def load_data(self):
        super().load_data()
        # reshape X's to 3 dims
        X_train_shape = self.X_train.shape
        X_test_shape = self.X_test.shape
        self.X_train = self.X_train.reshape((1, X_train_shape[0], X_train_shape[1]))
        self.X_test = self.X_test.reshape((1, X_test_shape[0], X_test_shape[1]))

        if self.classification:
            # one-hot encode when its classification
            self.y_train = to_categorical([ self.emotions2int[str(e)] for e in self.y_train ])
            self.y_test = to_categorical([ self.emotions2int[str(e)] for e in self.y_test ])
        
        # reshape labels
        y_train_shape = self.y_train.shape
        y_test_shape = self.y_test.shape
        if self.classification:
            self.y_train = self.y_train.reshape((1, y_train_shape[0], y_train_shape[1]))    
            self.y_test = self.y_test.reshape((1, y_test_shape[0], y_test_shape[1]))
        else:
            self.y_train = self.y_train.reshape((1, y_train_shape[0], 1))
            self.y_test = self.y_test.reshape((1, y_test_shape[0], 1))

    def train(self, override=False):

        # if model isn't created yet, create it
        if not self.model_created:
            self.create_model()

        # if the model already exists and trained, just load the weights and return
        # but if override is True, then just skip loading weights
        if not override:
            model_name = self._model_exists()
            if model_name:
                self.model.load_weights(model_name)
                self.model_trained = True
                if self.verbose > 0:
                    print("[*] Model weights loaded")
                return
        
        if not os.path.isdir("results"):
            os.mkdir("results")

        if not os.path.isdir("logs"):
            os.mkdir("logs")

        model_filename = self._get_model_filename()

        self.checkpointer = ModelCheckpoint(model_filename, save_best_only=True, verbose=1)
        self.tensorboard = TensorBoard(log_dir=os.path.join("logs", self.model_name))

        self.history = self.model.fit(self.X_train, self.y_train,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        validation_data=(self.X_test, self.y_test),
                        callbacks=[self.checkpointer, self.tensorboard],
                        verbose=self.verbose)
        
        self.model_trained = True
        if self.verbose > 0:
            print("[+] Model trained")

    def predict(self, audio_path):
        feature = extract_feature(audio_path, **self.audio_config).reshape((1, 1, self.input_length))
        if self.classification:
            prediction = self.model.predict(feature)
            prediction = np.argmax(np.squeeze(prediction))
            return self.int2emotions[prediction]
        else:
            return np.squeeze(self.model.predict(feature))

    def test_score(self):
        y_test = self.y_test[0]
        if self.classification:
            y_pred = self.model.predict(self.X_test)[0]
            y_pred = [np.argmax(y, out=None, axis=None) for y in y_pred]
            y_test = [np.argmax(y, out=None, axis=None) for y in y_test]
            return accuracy_score(y_true=y_test, y_pred=y_pred)
        else:
            y_pred = self.model.predict(self.X_test)[0]
            return mean_absolute_error(y_true=y_test, y_pred=y_pred)

    def train_score(self):
        y_train = self.y_train[0]
        if self.classification:
            y_pred = self.model.predict(self.X_train)[0]
            y_pred = [np.argmax(y, out=None, axis=None) for y in y_pred]
            y_train = [np.argmax(y, out=None, axis=None) for y in y_train]
            return accuracy_score(y_true=y_train, y_pred=y_pred)
        else:
            y_pred = self.model.predict(self.X_train)[0]
            return mean_absolute_error(y_true=y_train, y_pred=y_pred)


if __name__ == "__main__":

    rec = DeepEmotionRecognizer(emotions=["sad", "neutral", "happy" , "calm", "angry"],
                                epochs=300, verbose=0)
    rec.train(override=False)
    print("Test accuracy score:", rec.test_score() * 100, "%")

  