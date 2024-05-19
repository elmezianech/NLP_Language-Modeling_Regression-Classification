# NLP_Language-Modeling_Regression-Classification
This repository serves as a comprehensive exploration of Natural Language Processing (NLP) language models using the Sklearn library. It delves into both regression and classification tasks, utilizing various techniques and algorithms to analyze text data.

It consists of two parts: Language Modeling for regression tasks and Language Modeling for classification tasks. Each part involves cleaning, preprocessing the data, encoding it using different methods such as Word2Vec, and TF-IDF, training models with algorithms like SVR, Naive Bayes, Linear Regression, Decision Tree, and evaluating the models using standard metrics.

## Part 1: Language Modeling / Regression

This section focuses on regression tasks. It includes preprocessing steps such as tokenization, lemmatization, stop word removal, and more. The dataset used for regression tasks is obtained from this GitHub repository : https://github.com/dbbrandt/short_answer_granding_capstone_project/blob/master/data/sag/answers.csv. The code provided establishes an NLP pipeline for preprocessing, encodes data vectors using various methods, trains models with different algorithms, evaluates model performance using metrics like MSE and RMSE, and interprets the results obtained.

1. Data Preprocessing:

Loading data: The script loads a dataset from a CSV URL containing answer text and corresponding scores.

Cleaning Text: It cleans the text data by removing non-string values, converting text to lowercase, and eliminating HTML tags, URLs, numbers, punctuation marks, and stopwords. Lemmatization is also applied to reduce words to their base form. Word2Vec is then used to generate vector representations of the processed text.

2. Embedding:

Word2Vec: The Skip-Gram approach is used in Word2Vec to create numerical vector representations for each processed Tweet. These vectors capture semantic relationships between words.

Document Vectorization: Word2Vec creates numerical vector representations for each processed text document (answer) based on word relationships.

3. Model Training and Evaluation:

Splitting Data: The data is split into training and testing sets for model training and evaluation.

Model Selection: Four regression models are chosen: Support Vector Regression (SVR), Linear Regression, Decision Tree Regression, and Random Forest Regression.
Hyperparameter Tuning: GridSearchCV is employed to find the optimal hyperparameters for SVR, Decision Tree, and Random Forest models. 

Model Fitting: Each model is trained on the training data using the identified hyperparameters.

Performance Evaluation: The Mean Squared Error (MSE) is used as the evaluation metric during GridSearchCV to select the best hyperparameter combination. Lower MSE indicates better model performance.

4. Results and Interpretation:

The code outputs the best hyperparameters and corresponding CV (cross-validation) scores for each model.

#### Key Findings:
Random Forest achieved the lowest MSE, demonstrating the best performance in predicting the score based on the processed text data.

SVR with an RBF kernel performed well but fell slightly behind Random Forest.

Linear Regression showed the highest MSE, suggesting it might not be suitable for this dataset.

Decision Tree exhibited moderate performance, ranking between Random Forest and Linear Regression.

## Part 2: Language Modeling / Classification

This section focuses on classification tasks. It explores sentiment analysis techniques for classifying the sentiment (Irrelevant, Negative, Neutral, Positive) of entities mentioned in Tweets from this dataset : https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis. The code provided establishes an NLP pipeline for preprocessing, encodes data vectors using various methods, trains models with different algorithms, evaluates model performance using metrics like accuracy, recall and f1 score, and interprets the results obtained.

1. Data Preprocessing:

Data Loading: The script loads a dataset from a CSV URL containing Tweet IDs, entities, sentiment labels, and Tweet content.

Cleaning Text: It performs extensive cleaning on the Tweet content text data, including:
- Removing rows with non-string values.
- Converting text to lowercase for consistency.
- Eliminating HTML tags, URLs, numbers, and punctuation marks.
- Applying stemming to reduce words to their base forms (e.g., "running" becomes "run").
- Removing stopwords (common words like "the" and "a").
- Removing emojis.

2. Embedding:

Word2Vec: The CBOW (Continuous Bag-of-Words) approach is used in Word2Vec to create numerical vector representations for each processed Tweet. These vectors capture semantic relationships between words.

Document Vectorization: Each Tweet's vector representation is calculated as the average of the word vectors within the Tweet.

3. Model Training and Evaluation:

Model Selection: Three classification models are chosen: Logistic Regression, Random Forest, and AdaBoost.

Model Training: Each model is trained on the features (document vectors) and sentiment labels using the training data split.

Evaluation: The models' performance is evaluated on the testing data split using accuracy score and classification reports (precision, recall, F1-score) for each sentiment class.

4. Results and Interpretation:

Random Forest achieved the highest accuracy (around 73.84%) among the three models using Word2Vec embeddings.

5. Alternative Feature Engineering with TF-IDF:

TF-IDF Vectorizer: This approach creates features based on the word frequency-inverse document frequency (TF-IDF) of words within each Tweet. It emphasizes the importance of words that are frequent in a specific Tweet but rare across the entire dataset.

Random Forest with TF-IDF: A new pipeline is created combining TF-IDF vectorization and the Random Forest model. This is trained and evaluated on the same data splits.

#### Key Findings:
TF-IDF with Random Forest significantly outperformed CBOW Word2Vec (accuracy of approximately 90.65% vs. 73.84%).

TF-IDF captured more relevant features from the Tweets compared to Word2Vec embeddings in this case, it exhibited consistently high precision, recall, and F1-score across all classes.

CBOW might require further exploration and optimization for better performance on this specific sentiment analysis task.

Overall, TF-IDF with Random Forest proved to be the most effective approach for sentiment classification on this Twitter entity dataset.

## Conclusion :
This repository offers a thorough investigation into Natural Language Processing (NLP) techniques for both regression and classification tasks. By meticulously preparing the data, employing a range of encoding techniques, and training models with diverse algorithms, it highlights the effectiveness of varied approaches in addressing these tasks.
