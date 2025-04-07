import pandas as pd

#machine learning imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#plotting imports
import seaborn as sns
from matplotlib import pyplot as plt


def main(): 
    data = pd.read_csv("MushroomClassification\mushrooms.csv")
    #print(data)

    # Convert categorical data to numerical data using one-hot encoding
    data_encoded = pd.get_dummies(data)

    # Separate features and target
    x = data_encoded.drop(columns=["class_e", "class_p"])   # 'class' column is now split into 'class_e' and 'class_p'
    y = data_encoded["class_p"]                             # Class 'p' as target (poisonous or not) (1 or 0)   

    #Training the model (x, y, 20% of the data)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2) 
    
    #Making the model
    model = LogisticRegression(max_iter = 100)
    
    #Training the Model
    model.fit(x_train, y_train)

    #Predicting
    y_predictions = model.predict(x_test)
    
    #Evaluate
    accuracy = accuracy_score(y_test, y_predictions)
    accuracyPercent = str(accuracy * 100)
    print(f"accuracy for {model}: {accuracyPercent}%")

    # Plot confusion matrix
    conf_matrix = confusion_matrix(y_test, y_predictions)
    
    # Plot the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Edible', 'Poisonous'], yticklabels=['Edible', 'Poisonous'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


#Running main
if __name__ == "__main__":
    main()
    