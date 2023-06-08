from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW, AutoModelForSequenceClassification, AutoTokenizer,TFAutoModel,pipeline
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import torch

df2021 = pd.DataFrame([(k, v['time'], v['text'], v['isSad']) for k, v in data2021.items()], columns=['ID', 'Timestamp', 'Text', 'Label'])
df2022 = pd.DataFrame([(k, v['time'], v['text'], v['isSad']) for k, v in data2022.items()], columns=['ID', 'Timestamp', 'Text', 'Label'])
df = pd.concat([df2021, df2022], ignore_index=True)


def clean_empty_lists(row):
    if all([x == '' for x in row['Text']]):
        return np.nan
    else:
        return row
df_clean = df.apply(clean_empty_lists, axis=1).dropna()

df_sorted = df_clean.sort_values('Label', ascending=False)
# get all rows with Labels=1
df_label_1 = df_sorted[df_sorted['Label'] == 1]
# get the same amount of random samples from Labels=0
num_samples = len(df_label_1)
df_label_0 = df_sorted[df_sorted['Label'] == 0].sample(n=num_samples)
# conctenate the two DataFrames and shuffle the rows
df_concat = pd.concat([df_label_1, df_label_0]).sample(frac=1).reset_index(drop=True)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
# Generate BERT embeddings for each text in the dataset

df_concat['embeddings_bert'] = 0
huge_emb_lst = []
for i in range(336, len(df_concat)):
    emb_lst = []
    for text_piece in df_concat['Text'].iloc[i]:
        embedd = tokenizer(text_piece, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            outputs = model(**embedd)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        emb_lst.append(embeddings)
    # df_concat['embeddings'].iloc[i] = emb_lst
    huge_emb_lst.append(emb_lst)
df_concat['embeddings_bert'] = huge_emb_lst


# Load the pre-trained model
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
df_concat['embeddings_use'] = 0
huge_emb_lst = []
for i in range(len(df_concat)):
    emb_lst = []
    for text_piece in eval(df_concat['Text'].iloc[i]):
        if text_piece != '':
            embedd = embed([str(text_piece)])
            embedd = np.squeeze(embedd.numpy())
        else:
            embedd = np.zeros(512)
        emb_lst.append(embedd)
    # df_concat['embeddings'].iloc[i] = emb_lst
    huge_emb_lst.append(emb_lst)
df_concat['embeddings_use'] = huge_emb_lst



tokenizer = AutoTokenizer.from_pretrained("tae898/emoberta-large")
model = AutoModelForSequenceClassification.from_pretrained("tae898/emoberta-large")
emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
df_concat['emotions'] = 0
huge_emb_lst = []
for i in range(366, len(df_concat)):
    print('subject', i)
    emb_lst = []
    for text_piece in eval(df_concat['Text'].iloc[i]):
        embedd = emotion_classifier(text_piece[:512])
        emotions = np.array([[emotion['score'] for emotion in example] for example in embedd])
        emb_lst.append(emotions)
    # df_concat['embeddings'].iloc[i] = emb_lst
    huge_emb_lst.append(emb_lst)
df_concat['emotions'] = huge_emb_lst

df_concat.to_csv('df_concat.csv')
