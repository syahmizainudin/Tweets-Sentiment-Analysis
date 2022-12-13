![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=square-flat&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=square-flat&logo=Keras&logoColor=white)

# Sentiment Analysis on Tweets with LSTM
 Using LSTM to do NLP sentiment analysis on Tweets from Twitter.

## Usage
> Download the dataset from [here](https://www.kaggle.com/datasets/tariqsays/sentiment-dataset-with-1-million-tweets) and extract it into the dataset folder.

## Steps taken

Step 1 - Data loading
Step 2 - Data inspection
Step 3 - Data cleaning
Step 4 - Features selection
Step 5 - Data pre-preprocessing
Step 6 - Model development
Step 7 - Model evaluation
Step 8 - Model saving

## Visualization
<p align="center">
  <img src="resources/label_distributions.png" />
</p>
<p align='center'>A plot for the label distributions. The label in the dataset is mostly balanced between the 4 values.</p>
<p><br></p>

<p align="center">
  <img src="resources/language_distributions.png" />
</p>
<p align='center'>A plot for the language distributions. The data is is mostly made up of tweets in English.</p>
<p><br></p>

<p align="center">
  <img src="resources/model.png" />
</p>
<p align='center'>A plot for the structure of the model. The model consist of an Embedding layer, 2 layers of LSTM, a Dropout layer and a Dense layer as the classifier.</p>
<p><br></p>

<p align="center">
  <img src="resources/confusion_matrix.png" width=500/>
</p>
<p align='center'>Confusion matrix from the model's predictions normalized to 1 for easier analysis.</p>
<p><br></p>

<p align="center">
  <img src="resources/classification_report.png" width=500/>
</p>
<p align='center'>Classifcation report from the model's predictions. The model scored 95% in accuracy and between 94%-96% in F1-score.</p>

## Acknowledgement
Thanks to [Muhammad Tariq](https://www.kaggle.com/datasets/tariqsays/sentiment-dataset-with-1-million-tweets) from Kaggle for the dataset!
