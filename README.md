# Sentiment-analysis-on-IMDB-movie-reviews

# The notebooks contain codes for developing machine learning models to classify the sentiment of IMDB movie reviews. The notebooks are self-explanatory with comments.

IMDB_Movie_Review_Sentiment_explore.ipynb Notebook outline
-----------------------------------------------------------
1. Import Libraries

2. Setting up Google Colab (Reviewer can ignore this)

3. Exploring the dataset

4. Preprocessing

5. Data Visualization after Preprocessing

6. Generating Bag of Words and TF-IDF features for Baseline models

7. Training the Baseline Models (Naive Bayes and Logictic Regression) - achieved upto 74% accuracy

8. Improvement over the beaseline model: BiLSTM model with pre-trained GloVe word embeddings

        8.1 Network selection using Keras Hyperparameter training

        8.2 Training the Best Model (upto ~88% accuracy)

        8.3 Visualizing training History and testing the model

9. Evaluation

10. End Remarks


XLNetforIMDB.ipynb Notebook outline
------------------------------------
XLNet[1] is one of the state-of-the-art transformer based models for IMDB movie review sentiment classification which achieved 96.2% accuracy. In this notebook, I trained a XLNet model using xlnet-base-cased model from Hugging Face Transformers (https://github.com/huggingface/transformers) and Simple Transformers (https://simpletransformers.ai/docs/tips-and-tricks/#using-early-stopping) library.

Notes:
1. Due to GPU limitation in Colab, I used only a xlnet base model and not the large version of the model
2. The xlnet model is trained for only 1 epoch
3. No data preprocessing is done for the XLNet model
4. XLNet achieved 91.7% accuracy on the test data.



[1] Yang, Z., Dai, Z., Yang, Y., Carbonell, J., Salakhutdinov, R. R., & Le QV, X. (2021). generalized autoregressive pretraining for language understanding; 2019. Preprint at https://arxiv. org/abs/1906.08237 Accessed June, 21.

[2] https://news.machinelearning.sg/posts/sentiment_analysis_on_movie_reviews_with_xlnet/
