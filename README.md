# IMDB-movie-review-sentiment-classification-using-CNN-and-keras-colab
Dataset Information IMDB dataset contains 50K movie reviews for natural language processing i.e. for binary sentiment classification. The dataset contains two columns - review and sentiment to perform the sentimental analysis.  Problem Statement Correctly classify the positive and negative sentiments for IMDB reviews.
IMDB Movie Review Sentiment Classification using CNN and Keras/Colab
This project uses convolutional neural networks (CNNs) and the Keras library to perform sentiment classification on the IMDB movie review dataset. The goal is to predict whether a movie review is positive or negative based on the text of the review.

Data
The IMDB movie review dataset contains 50,000 movie reviews that have been labeled as either positive or negative. The dataset is split into training and testing sets, with 25,000 reviews in each set. The reviews are already preprocessed and converted to sequences of integers, where each integer represents a word in the review.

Architecture
The model is built using a 1D convolutional neural network, which takes the sequence of integers as input and learns to classify the sentiment based on the patterns in the data. The model consists of an embedding layer, followed by multiple convolutional and pooling layers, and finally a dense output layer with a sigmoid activation function.

Results
The model achieves an accuracy of ** 88.54%** on the testing set, demonstrating its effectiveness at performing sentiment classification on movie reviews.

Usage
The code for this project is written in Python using the Keras library and was run on Google Colab. To run the code, simply open the final_pattern_project.ipynb notebook in Colab or Jupyter and follow the instructions in the notebook.

Dependencies
This project requires the following libraries:

Keras
NumPy
Matplotlib
Pandas
Scikit-learn
These libraries can be installed using pip or conda.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
This project was inspired by the Keras documentation and the IMDB movie review dataset. Special thanks to the Keras and Google Colab teams for their excellent tools and resources.
