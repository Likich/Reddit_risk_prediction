from tensorflow.keras.layers import Input, Dense, LSTM, Lambda
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Define the number of classes
NUM_CLASSES = 2
# Define the GRU parameters
LSTM_HIDDEN_DIM = 256
GRU_HIDDEN_DIM = 256
MAX_SEQ_LENGTH=1265
DROPOUT_RATE = 0.2

EmbeddingInput = Input(shape=(MAX_SEQ_LENGTH, 768), name="embeddings")
EmotionInput = Input(shape=(MAX_SEQ_LENGTH, 7), name="EmotionInput")
TimeInput = Input(shape=(MAX_SEQ_LENGTH,), name="times")
decay_layer = Lambda(
    lambda t: tf.math.exp(-(t - tf.roll(t, shift=1, axis=1)) / 86400),
    name="decay_layer",
)(TimeInput)
decay_layer_2 = tf.expand_dims(decay_layer, axis=-1, name="decay_layer_2")
lstm_layer = LSTM(
    LSTM_HIDDEN_DIM, dropout=DROPOUT_RATE, return_sequences=True, name="lstm"
)(EmbeddingInput, mask=EmbeddingInput._keras_mask)
lstm_layer_emotion = LSTM(
    LSTM_HIDDEN_DIM, dropout=DROPOUT_RATE, return_sequences=True, name="lstm2"
)(EmotionInput, mask=EmotionInput._keras_mask)
multiply = tf.keras.layers.Multiply(name="multiply")(
    [lstm_layer, lstm_layer_emotion, decay_layer_2]
)
flatten_layer = tf.keras.layers.Flatten(name="flatten")(multiply)
outputs = Dense(NUM_CLASSES, activation="softmax")(flatten_layer)
model = Model(inputs=[EmbeddingInput, TimeInput, EmotionInput], outputs=outputs)
optimizer = Adam(learning_rate=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
