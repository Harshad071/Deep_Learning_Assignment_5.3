# Amazon Product Review Sentiment Analysis using LSTM

This project focuses on **sentiment analysis** of Amazon product reviews using a **Long Short-Term Memory (LSTM)** model. The goal of the project is to predict the sentiment of a product review — whether it's **positive** or **negative** — based on the text of the review. The dataset consists of **Amazon product reviews**, which includes text feedback provided by users along with a score rating from 1 to 5.

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Dependencies](#dependencies)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Architecture](#model-architecture)
6. [Training and Evaluation](#training-and-evaluation)
7. [Results](#results)
8. [Usage](#usage)
9. [Conclusion](#conclusion)

## Overview

The project uses an **LSTM (Long Short-Term Memory)** model for sentiment analysis on Amazon product reviews. The reviews are classified into **positive** and **negative** sentiment classes based on their score ratings. The sentiment classification is performed after preprocessing the text data, tokenizing it into numerical sequences, and padding them to a uniform length. The LSTM model is then trained to predict the sentiment of new, unseen reviews.

The model's evaluation is done using standard metrics such as **accuracy**, **precision**, **recall**, and **F1-score**. Additionally, a **word cloud** is generated to visualize the most frequent words in the reviews.

## Dataset

The dataset used in this project is the **Amazon Product Reviews** dataset, available on Kaggle. You can download it [here](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews).

### Dataset Features:
- **Id**: Unique identifier for each review.
- **ProductId**: Unique identifier for each product.
- **UserId**: Unique identifier for the user.
- **ProfileName**: Name of the user profile.
- **HelpfulnessNumerator**: Number of users who found the review helpful.
- **HelpfulnessDenominator**: Total number of users who voted for the review.
- **Score**: Rating score given to the product (from 1 to 5).
- **Time**: Timestamp when the review was posted.
- **Summary**: A brief summary of the review.
- **Text**: Full text of the review.

### Sentiment Labeling:
- Reviews with **Score ≥ 4** are considered **Positive** (labeled as 1).
- Reviews with **Score ≤ 2** are considered **Negative** (labeled as 0).
- **Score 3** is excluded for this binary classification as neutral sentiment.

## Dependencies

To run this project, you need to install the following Python libraries:

- **TensorFlow**: For building the deep learning model using LSTM.
- **Keras**: High-level neural networks API built on top of TensorFlow.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib**: For creating visualizations and plots.
- **Seaborn**: For advanced visualizations.
- **Scikit-learn**: For data preprocessing and evaluation metrics.

You can install these libraries by running the following command:

pip install tensorflow pandas numpy matplotlib seaborn scikit-learn

## Results

After training the LSTM model on the Amazon Product Reviews dataset, the following results were achieved:

- **Accuracy**: 96.06%
- **Precision**: 96.06%
- **F1-score**: 96.06%

These results indicate that the model performs well in classifying reviews into positive and negative sentiment categories. The confusion matrix and classification report further validate the model's performance.

## Usage

To use this project, follow these steps:

1. **Download the dataset** from [Amazon Product Reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews).
2. **Preprocess the data** by cleaning the reviews and tokenizing them.
3. **Train the LSTM model** using the provided architecture.
4. **Evaluate the model** using accuracy and classification metrics.
5. **Generate visualizations** such as confusion matrix and word cloud.

## Conclusion

This project demonstrates the application of **LSTM (Long Short-Term Memory)** models for **sentiment analysis** on Amazon product reviews. The model performs well with a high accuracy of 96.06% in classifying reviews as either **positive** or **negative**. Further improvements can be made by fine-tuning the model, handling neutral reviews, or utilizing pre-trained word embeddings.
