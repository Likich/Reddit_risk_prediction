from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Attention
import keras.backend as K

# Define the time decay function
def time_decay(epoch):
    lrate = INIT_LR * pow(DECAY_FACTOR, np.floor((1 + epoch) / DECAY_EPOCHS))
    return lrate

class attention(Layer):
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight", shape=(input_shape[-1], 1), initializer="normal"
        )
        self.b = self.add_weight(
            name="att_bias", shape=(input_shape[1], 1), initializer="zeros"
        )
        super(attention, self).build(input_shape)

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
        return super(attention, self).get_config()


train_emobed_data = np.squeeze(train_emobed_data)
val_data_emobed = np.squeeze(val_data_emobed)
test_data_emobed = np.squeeze(test_data_emobed)
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
attention_layer = attention(name="attention")
att_out, attention_weights = attention_layer(multiply)
attention_scores = tf.keras.layers.Flatten(name="attention_scores")(attention_weights)
outputs = Dense(NUM_CLASSES, activation="softmax", name="output_real")(att_out)
model = Model(inputs=[EmbeddingInput, TimeInput, EmotionInput], outputs=[outputs, attention_scores])
optimizer = Adam(learning_rate=INIT_LR)
model.compile(
        loss={
            "output_real": "binary_crossentropy",
            "attention_scores": "mean_absolute_error",
        },
        optimizer=optimizer,
        metrics={"output_real": ["accuracy"]},
    )
lr_scheduler = LearningRateScheduler(time_decay)



history = model.fit(
    [train_data, train_timestamps_padded, train_emobed_data],
    {"output_real": train_labels, "attention_scores": np.zeros((392, 1265))},
    validation_data=(
        [val_data, val_timestamps_padded, val_data_emobed],
        {"output_real": val_labels, "attention_scores": np.zeros((49, 1265))},
    ),
    epochs=10,
    batch_size=32,
    callbacks=[lr_scheduler],
    shuffle=False,
)

test_preds = model.predict([test_data, test_timestamps_padded, test_data_emobed])
test_preds = np.argmax(test_preds[0], axis=1)
test_f1_score = f1_score(
        np.argmax(test_labels, axis=1), test_preds, average="macro"
    )
test_auroc = roc_auc_score(np.argmax(test_labels, axis=1), test_preds)
test_auprc = average_precision_score(np.argmax(test_labels, axis=1), test_preds)
test_accuracy = accuracy_score(np.argmax(test_labels, axis=1), test_preds)
test_recall = recall_score(
        np.argmax(test_labels, axis=1), test_preds, average="macro"
    )
test_precision = precision_score(
        np.argmax(test_labels, axis=1), test_preds, average="macro"
    )
print("Test AUROC:", test_auroc)
print("Test AUPRC:", test_auprc)
print("Test Accuracy:", test_accuracy)
print("Test Recall:", test_recall)
print("Test Precision:", test_precision)
print("Test F1 Score:", test_f1_score)
