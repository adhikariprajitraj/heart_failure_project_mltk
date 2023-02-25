import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tkinter as tk

# Load the dataset
df = pd.read_csv('heart.csv')

# Perform one-hot encoding on the 'RestingECG' and 'ST_Slope' columns
df = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'ST_Slope', 'RestingECG', 'ExerciseAngina'])


# Split the dataset into features and target variable
X = df.drop(columns=['HeartDisease']).values
y = df['HeartDisease'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

# Train an SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Create a tkinter window for user input
window = tk.Tk()
window.title("Heart Disease Predictor")

# Create labels and entry boxes for each feature
tk.Label(window, text="Age").grid(row=0)
age_entry = tk.Entry(window)
age_entry.grid(row=0, column=1)

tk.Label(window, text="Sex (0 = Female, 1 = Male)").grid(row=1)
sex_entry = tk.Entry(window)
sex_entry.grid(row=1, column=1)

tk.Label(window, text="Chest Pain Type").grid(row=2)
chest_pain_entry = tk.Entry(window)
chest_pain_entry.grid(row=2, column=1)

tk.Label(window, text="Resting Blood Pressure").grid(row=3)
resting_bp_entry = tk.Entry(window)
resting_bp_entry.grid(row=3, column=1)

tk.Label(window, text="Cholesterol").grid(row=4)
cholesterol_entry = tk.Entry(window)
cholesterol_entry.grid(row=4, column=1)

tk.Label(window, text="Fasting Blood Sugar (0 = False, 1 = True)").grid(row=5)
fasting_bs_entry = tk.Entry(window)
fasting_bs_entry.grid(row=5, column=1)

tk.Label(window, text="Max Heart Rate Achieved").grid(row=6)
max_hr_entry = tk.Entry(window)
max_hr_entry.grid(row=6, column=1)

tk.Label(window, text="Exercise-Induced Angina (0 = False, 1 = True)").grid(row=7)
exercise_angina_entry = tk.Entry(window)
exercise_angina_entry.grid(row=7, column=1)

tk.Label(window, text="ST Depression Induced by Exercise Relative to Rest").grid(row=8)
oldpeak_entry = tk.Entry(window)
oldpeak_entry.grid(row=8, column=1)

# Create a function to predict heart disease based on the user's input
def predict():
    # Get the user's input and convert it to a numpy array
    input_data = np.array([
        float(age_entry.get()),
        float(sex_entry.get()),
        float(chest_pain_entry.get()),
        float(resting_bp_entry.get()),
        float(cholesterol_entry.get()),
        float(fasting_bs_entry.get()),
        float(resting_ecg_var.get()),
        float(max_hr_entry.get()),
        float(exercise_angina_entry.get()),
        float(oldpeak_entry.get()),
        float(st_slope_var.get())]).reshape(1, -1)
    # Normalize the input data using the scaler from the training set
    input_data = scaler.transform(input_data)

    # Use the logistic regression model to predict heart disease
    lr_prediction = lr_model.predict(input_data)[0]

    # Use the SVM model to predict heart disease
    svm_prediction = svm_model.predict(input_data)[0]

    # Display the predicted heart disease in the output label
    if lr_prediction == 0 and svm_prediction == 0:
        output_label.config(text="No Heart Disease")
    elif lr_prediction == 1 and svm_prediction == 1:
        output_label.config(text="Heart Disease")
    else:
        output_label.config(text="Inconclusive")

predict_button = tk.Button(window, text="Predict", command=predict)
predict_button.grid(row=9, column=1)

tk.Label(window, text="Resting ECG").grid(row=10)
resting_ecg_var = tk.StringVar(value="0")
tk.Radiobutton(window, text="Normal", variable=resting_ecg_var, value="0").grid(row=11)
tk.Radiobutton(window, text="ST-T Wave Abnormality", variable=resting_ecg_var, value="1").grid(row=12)
tk.Radiobutton(window, text="Left Ventricular Hypertrophy", variable=resting_ecg_var, value="2").grid(row=13)

tk.Label(window, text="ST Slope").grid(row=14)
st_slope_var = tk.StringVar(value="0")
tk.Radiobutton(window, text="Upsloping", variable=st_slope_var, value="0").grid(row=15)
tk.Radiobutton(window, text="Flat", variable=st_slope_var, value="1").grid(row=16)
tk.Radiobutton(window, text="Downsloping", variable=st_slope_var, value="2").grid(row=17)


output_label = tk.Label(window, text="")
output_label.grid(row=18, columnspan=2)

window.mainloop()