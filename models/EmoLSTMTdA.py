from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, LSTM, Lambda, Multiply
from tensorflow.keras.models import Model
import tensorflow as tf

# Define the number of classes
NUM_CLASSES = 2
# Define the GRU parameters
LSTM_HIDDEN_DIM = 256
GRU_HIDDEN_DIM = 256
MAX_SEQ_LENGTH=1265
DROPOUT_RATE = 0.2

# Custom attention layer
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight", shape=(input_shape[-1], 1), initializer="normal"
        )
        self.b = self.add_weight(
            name="att_bias", shape=(input_shape[1], 1), initializer="zeros"
        )
        super(Attention, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        attention_weights = at  # Return attention weights
        return K.sum(output, axis=1), attention_weights

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        return super(Attention, self).get_config()

# Define the inputs
EmbeddingInput = Input(shape=(MAX_SEQ_LENGTH, 768), name="embeddings")
EmotionInput = Input(shape=(MAX_SEQ_LENGTH, 7), name="EmotionInput")
TimeInput = Input(shape=(MAX_SEQ_LENGTH,), name="times")

# Time decay layer
decay_layer = Lambda(
    lambda t: tf.math.exp(-(t - tf.roll(t, shift=1, axis=1)) / 86400),
    name="decay_layer",
)(TimeInput)
decay_layer_2 = tf.expand_dims(decay_layer, axis=-1, name="decay_layer_2")

# LSTM layers
lstm_layer = LSTM(
    LSTM_HIDDEN_DIM, dropout=DROPOUT_RATE, return_sequences=True, name="lstm"
)(EmbeddingInput, mask=EmbeddingInput._keras_mask)
lstm_layer_emotion = LSTM(
    LSTM_HIDDEN_DIM, dropout=DROPOUT_RATE, return_sequences=True, name="lstm2"
)(EmotionInput, mask=EmotionInput._keras_mask)

# Multiply the LSTM outputs with the decay layer
multiply = Multiply(name="multiply")([lstm_layer, lstm_layer_emotion, decay_layer_2])

# Attention layer
attention_layer = Attention(name="attention")
att_out, attention_weights = attention_layer(multiply)
attention_scores = tf.keras.layers.Flatten(name="attention_scores")(attention_weights)

# Output layer
outputs = Dense(NUM_CLASSES, activation="softmax", name="output_real")(att_out)

# Define the model
model = Model(inputs=[EmbeddingInput, TimeInput, EmotionInput], outputs=[outputs, attention_scores])
