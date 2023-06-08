import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from embeddings import huge_emb_lst

df_concat = pd.read_csv('df_concat.csv')


# Iterate over each row of the DataFrame
for index, row in df_concat.iterrows():
    # Get the list of timestamps and the list of text for this row
    timestamps = row['Timestamp']
    text_list = row['Text']
    cleaned_text_list = [text for text in text_list if text != '']
    if len(cleaned_text_list) == 0:
        continue
    cleaned_timestamps = [timestamps[i] for i in range(len(text_list)) if text_list[i] != '']
    df_concat.at[index, 'Text'] = cleaned_text_list
    df_concat.at[index, 'Timestamp'] = cleaned_timestamps
    

# Split the data into training and validation sets
train_texts, val_test_texts, train_labels, val_test_labels, train_timestamps, val_test_timestamps = train_test_split(
    df_concat['emobeds'].values,
    df_concat['Label'].values,
    df_concat['Timestamp'].values,
    test_size=0.2,
    random_state=42
)
val_texts, test_texts, val_labels, test_labels, val_timestamps, test_timestamps = train_test_split(
    val_test_texts,
    val_test_labels,
    val_test_timestamps,
    train_size=0.5
)

NUM_CLASSES = 2
MAX_SEQ_LENGTH = max(len(x) for x in train_texts[:][:])

# Convert the labels to one-hot vectors
train_labels = tf.keras.utils.to_categorical(train_labels, NUM_CLASSES)
val_labels = tf.keras.utils.to_categorical(val_labels, NUM_CLASSES)
test_labels = tf.keras.utils.to_categorical(test_labels, NUM_CLASSES)

# Pad the sequences to the same length
train_data = tf.keras.preprocessing.sequence.pad_sequences(train_texts, maxlen=MAX_SEQ_LENGTH, dtype='float32', padding='post', truncating='post')
val_data = tf.keras.preprocessing.sequence.pad_sequences(val_texts, maxlen=MAX_SEQ_LENGTH, dtype='float32', padding='post', truncating='post')
test_data = tf.keras.preprocessing.sequence.pad_sequences(test_texts, maxlen=MAX_SEQ_LENGTH, dtype='float32', padding='post', truncating='post')


# Convert the timestamps to datetime objects
train_data_timestamps = []
for ts_list in train_timestamps:
    ts_list = [int(datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').timestamp()) for ts in ts_list]
    train_data_timestamps.append(ts_list)

val_data_timestamps = []
for ts_list in val_timestamps:
    ts_list = [int(datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').timestamp()) for ts in ts_list]
    val_data_timestamps.append(ts_list)

test_data_timestamps = []
for ts_list in test_timestamps:
    ts_list = [int(datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').timestamp()) for ts in ts_list]
    test_data_timestamps.append(ts_list)

train_timestamps_padded = tf.keras.preprocessing.sequence.pad_sequences(train_data_timestamps, padding='post', maxlen=MAX_SEQ_LENGTH)
val_timestamps_padded = tf.keras.preprocessing.sequence.pad_sequences(val_data_timestamps, padding='post', maxlen=MAX_SEQ_LENGTH)
test_timestamps_padded = tf.keras.preprocessing.sequence.pad_sequences(test_data_timestamps, padding='post', maxlen=MAX_SEQ_LENGTH)


np.save('train_data_emobed', np.array(train_data))
np.save('val_data_emobed', np.array(val_data))
np.save('test_data_emobed', np.array(test_data))
np.save('train_timestamps_padded_emobed', np.array(train_timestamps_padded))
np.save('val_timestamps_padded_emobed', np.array(val_timestamps_padded))
np.save('test_timestamps_padded_emobed', np.array(test_timestamps_padded))
np.save('train_labels_emobed', np.array(train_labels))
np.save('val_labels_emobed', np.array(val_labels))
np.save('test_labels_emobed', np.array(test_labels))
np.save('emoberta_full', huge_emb_lst)


# normalizing time inputs was super important!!
max_time = 1./np.max(train_timestamps_padded)
train_timestamps_padded = (train_timestamps_padded*max_time).astype(np.float32)
val_timestamps_padded = (val_timestamps_padded*max_time).astype(np.float32)
test_timestamps_padded = (test_timestamps_padded*max_time).astype(np.float32)