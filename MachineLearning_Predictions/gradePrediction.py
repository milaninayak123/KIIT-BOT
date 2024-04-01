import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def getGrade(marks):


    data = pd.read_csv("ML_Datasets/student.csv", sep=",")

    data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
    predict = "G3"
    X = np.array(data.drop(columns=[predict]))
    y = np.array(data[predict])

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(x_train, y_train)  # Train the model

    prediction = rf_model.predict([marks])[0]


    return prediction
