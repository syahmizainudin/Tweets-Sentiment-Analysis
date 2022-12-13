# %%
from tensorflow import keras
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense, Embedding
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import pandas as pd
import numpy as np
import os, re, datetime, json, pickle

# %% 1. Data loading
DATASET_PATH = os.path.join(os.getcwd(), 'dataset')

df = pd.read_csv(os.path.join(DATASET_PATH, 'dataset.csv'))

# %% 2. Data inspection
df.head(10)
df.tail(10)

df.info()
df.describe()

df.isna().sum() # 23 NaN value in Language column
df.duplicated().sum() # 4974 duplicated data

df['Language'].value_counts()
df['Label'].value_counts()

df['Text'][12937] # Twitter handle and emoji need to be removed
df['Text'][123456] # Links need to be removed

# Plot the distribution of the target, Label
plt.figure(figsize=(5,5))
sns.displot(df['Label'])
plt.title('Distributions of Label')
plt.show()

# Plot the distribution of the Language for Language where there is a minimum of 1000 data
filter = df['Language'].value_counts() > 1000
language_list = [x for x,y in zip(filter.index, filter.values) if y==True]
plt.figure(figsize=(5,5))
sns.displot(df['Language'].where(df['Language'].isin(language_list)))
plt.title('Distributions of Language')
plt.show()

# Plot a matrix of missing data
msno.matrix(df)

# %% 3. Data cleaning
# Dropping rows with NaN value as the amount is so small compared to the amount of data
df_drop = df.dropna(axis=0)
df_drop.isna().sum()

# Dropping duplicates
df_drop = df_drop.drop_duplicates()
df_drop.duplicated().sum()

# Function to clean data of anomalies
def clean_text(text):
    regex = '@[\w]+|http[\S]+|[^\w #]'
    return re.sub(regex, ' ', text).lower()

df_drop['Text'] = df_drop['Text'].apply(clean_text)

# Check if the text had been cleaned
print(df_drop['Text'][12937])
print(df_drop['Text'][123456])

# %% 4. Features inspection
# Selecting only tweets in English as it has the most amount of data
df_drop = df_drop[df_drop['Language'] == 'en']

# Define features and targets
features = df_drop['Text']
targets = df_drop['Label']

# Summary for data in text
np.sum(features.str.split().str.len())
np.mean(features.str.split().str.len())
np.median(features.str.split().str.len())
np.max(features.str.split().str.len())

# %% 5. Data pre-processing
# Tokenization
num_words = 300 
oov_token = '<OOV>'

tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(features)
features_sequence = tokenizer.texts_to_sequences(features)

# Padding + truncating
features_sequence = pad_sequences(features_sequence, maxlen=40, padding='post', truncating='post')

# Expand the dimension of both features and targets
features = np.expand_dims(features, -1)
targets = np.expand_dims(targets, -1)

# Encode the targets with OneHotEncoder
ohe = OneHotEncoder(sparse=False)
targets_encoded = ohe.fit_transform(targets)

# Train-test split
SEED = 12345
X_train, X_test, y_train, y_test = train_test_split(features_sequence, targets_encoded, test_size=0.2, random_state=SEED)

# %% Model development
# Define Sequential model layer
embedding_dim = 64

model = Sequential()
model.add(Embedding(num_words, embedding_dim))
model.add(LSTM(embedding_dim, return_sequences=True))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(y_train.shape[-1], activation='softmax'))

# Model summary
model.summary()
plot_model(model, show_shapes=True, show_layer_names=True, to_file=os.path.join(os.getcwd(), 'resources', 'model.png'))

# Model compiling
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='acc')

# Callbacks
LOG_DIR = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = TensorBoard(log_dir=LOG_DIR)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
es = EarlyStopping(monitor='val_loss', patience=3)

# %%
# Model training
EPOCHS = 10
BATCH_SIZE = 128

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[tb, reduce_lr, es])

# %% Model evaluation
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Metrics
print('Confusion matrix:\n', confusion_matrix(y_true, y_pred, normalize='true'))
print('Classification report:\n', classification_report(y_true, y_pred))

# %% Model saving
# Save the tokenizer
with open('tokenizer.json', 'w') as f:
    json.dump(tokenizer.to_json(), f)

# Save the encoder
with open('encoder.pkl', 'wb') as f:
    pickle.dump(ohe, f)

# Save the model
model.save('tweet-sentiment-analysis.h5')

# %%
