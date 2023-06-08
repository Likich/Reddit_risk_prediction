import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import  roc_auc_score, f1_score
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, recall_score, precision_score
from model import model
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from preprocess import train_data, train_timestamps_padded, test_data, test_timestamps_padded, val_data, val_timestamps_padded

# Define the learning rate decay parameters
INIT_LR = 0.001
DECAY_FACTOR = 0.1
DECAY_EPOCHS = 10

train_data = np.load('train_data.npy')
val_data = np.load('val_data.npy')
test_data = np.load('test_data.npy')
train_data_use = np.load('train_data_use.npy')
val_data_use = np.load('val_data_use.npy')
test_data_use = np.load('test_data_use.npy')
train_timestamps_padded = np.load('train_timestamps_padded.npy')
val_timestamps_padded = np.load('val_timestamps_padded.npy')
test_timestamps_padded = np.load('test_timestamps_padded.npy')
train_labels = np.load('train_labels.npy')
val_labels = np.load('val_labels.npy')
test_labels = np.load('test_labels.npy')
emoberta = np.load('emoberta_full.npy', allow_pickle=True)
emoberta_padded = tf.keras.preprocessing.sequence.pad_sequences(emoberta, padding='post', maxlen=1265)

np.random.seed(0)

data = np.vstack((train_data, val_data, test_data))
data_use = np.vstack((train_data_use, val_data_use, test_data_use))
time = np.vstack((train_timestamps_padded, val_timestamps_padded, test_timestamps_padded))
labels = np.vstack((train_labels, val_labels, test_labels))
# emobed = np.vstack((train_data_emobed, val_data_emobed, test_data_emobed))

train_data, val_test_data, train_emobed_data, val_test_emobed_data, train_labels, val_test_labels, train_timestamps_padded, val_test_timestamps_padded = train_test_split(
    data,
    emoberta_padded,
    labels,
    time,
    test_size=0.2,
    random_state=5
)
val_data, test_data, val_labels, test_labels, val_timestamps_padded, test_timestamps_padded, val_data_emobed, test_data_emobed = train_test_split(
    val_test_data,
    val_test_labels,
    val_test_timestamps_padded,
    val_test_emobed_data,
    train_size=0.5,
    random_state=5
)

train_data_use, val_test_data_use = train_test_split(
    data_use,
    test_size=0.2,
    random_state=0
)
val_data_use, test_data_use = train_test_split(
    val_test_data_use,
    train_size=0.5,
    random_state=0
)

# Define the time decay function
def time_decay(epoch):
    lrate = INIT_LR * pow(DECAY_FACTOR, np.floor((1 + epoch) / DECAY_EPOCHS))
    return lrate

# Compile the model
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

# Fit the model
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

# Evaluate on test data
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