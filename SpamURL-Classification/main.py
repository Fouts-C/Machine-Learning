# dataset used: https://www.kaggle.com/datasets/shivamb/spam-url-prediction

#imports here
import pandas as pd
import time 
from sklearn.preprocessing import LabelEncoder
#model training imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#evaluation imports
from sklearn.metrics import accuracy_score, confusion_matrix
#plotting imports
import seaborn as sns
from matplotlib import pyplot as plt

def main():

    data = pd.read_csv("SpamEmailClassification/url_spam_classification.csv")

    our_data = pd.read_csv("SpamEmailClassification/our_url.csv")           # Our links
    #our_data = pd.read_csv("SpamEmailClassification/synthetic_data.csv")   # Synthetic data
    #our_data = pd.read_csv("SpamEmailClassification/single_url.csv")       # Single link testing

    useGeneratedURLS = True
    if (useGeneratedURLS):
        # Mark the training and testing data
        data['is_train'] = True
        our_data['is_train'] = False
        data = pd.concat([data, our_data], ignore_index=True)

    # Convert URLs to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    x = vectorizer.fit_transform(data['url'])

    # Encode labels (spam or not spam)
    encoder = LabelEncoder()
    y = encoder.fit_transform(data['is_spam'])

    #split data into training and testing
    if (useGeneratedURLS):
        x_train = x[data['is_train']]
        y_train = y[data['is_train']]

        x_test = x[~data['is_train']]
        y_test = y[~data['is_train']]

    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2) 

    #choose a model (start with logistic regression)
    model = LogisticRegression(max_iter= 500, n_jobs= -1 )

    #fit the model
    model.fit(x_train, y_train)

    #generate predictions
    y_predictions = model.predict(x_test)

    #evaluate predictions
    accuracy = accuracy_score(y_test, y_predictions)
    accuracyPercent = str(accuracy * 100)
    print(f"accuracy for {model}: {accuracyPercent}%")

    # Plot confusion matrix
    conf_matrix = confusion_matrix(y_test, y_predictions)
    
    # Plot the confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    

#Running main
if __name__ == "__main__":
    main()