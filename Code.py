import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# Loading Data from fraudTrain.csv and fraudTest.csv in pandas DataFrame
# Since Data is not in the form of csv UTF-8 Format, we convert it into ISO-8859-1 format
train_data = pd.read_csv("fraudTrain.csv", encoding="ISO-8859-1")
test_data = pd.read_csv("fraudTest.csv", encoding="ISO-8859-1")

# Define the text columns you want to convert to TF-IDF
text_columns = ['trans_date_trans_time', 'merchant', 'category', 'first', 'last', 'street', 'city', 'state', 'job', 'dob', 'trans_num']

# Create TF-IDF vectorizers for each text column
tfidf_vectorizers = {col: TfidfVectorizer() for col in text_columns}

# Initialize a list to store TF-IDF data
X_train_tfidf = []

# Iterate through text columns for TF-IDF transformation
for col in text_columns:
    try:
        tfidf_data = tfidf_vectorizers[col].fit_transform(train_data[col])
        X_train_tfidf.append(tfidf_data)
    except ValueError as e:
        print(f"Skipping column '{col}' due to error: {e}")

# Initialize a list to store TF-IDF data for test data
X_test_tfidf = []

# Iterate through text columns for TF-IDF transformation for test data
for col in text_columns:
    try:
        tfidf_data = tfidf_vectorizers[col].transform(test_data[col])
        X_test_tfidf.append(tfidf_data)
    except ValueError as e:
        print(f"Skipping column '{col}' due to error: {e}")

# Convert 'gender' column to numeric values (0 for 'M', 1 for 'F')
train_data['gender'] = train_data['gender'].apply(lambda x: 0 if x == 'M' else 1)
test_data['gender'] = test_data['gender'].apply(lambda x: 0 if x == 'M' else 1)

# List of columns to include in X_train_final and X_test_final
included_columns = ['cc_num', 'amt', 'zip', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long', 'gender']

# Concatenate TF-IDF features with the original features
X_train_final = hstack([train_data[included_columns].values.astype('float64')] + X_train_tfidf, format='csr')
X_test_final = hstack([test_data[included_columns].values.astype('float64')] + X_test_tfidf, format='csr')

# Appointing Features and Target
Y_train = train_data['is_fraud']
Y_test = test_data['is_fraud']

# Convert Y_train and Y_test values to integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Training the model
model = LogisticRegression(solver='liblinear', max_iter=1000)

# Training the Logistic Regression Model with the training data
model.fit(X_train_final, Y_train)

# Prediction on training data
Prediction_on_Training_data = model.predict(X_train_final)

# Accuracy Score on Training_data
accuracy_on_Training_data = accuracy_score(Y_train, Prediction_on_Training_data)
print("The accuracy score on Training Data is: " + str(accuracy_on_Training_data * 100) + "%")

# Prediction on test data
Prediction_on_Test_data = model.predict(X_test_final)

# Accuracy Score on Test_data
accuracy_on_Test_data = accuracy_score(Y_test, Prediction_on_Test_data)
print("The accuracy score on Test Data is: " + str(accuracy_on_Test_data * 100) + "%")

# Define your custom input data as a dictionary
custom_input_data = {
    'cc_num': [123456789],
    'amt': [100.0],
    'zip': [12345],
    'lat': [40.0],
    'long': [-75.0],
    'city_pop': [10000],
    'unix_time': [1612458965],
    'merch_lat': [40.1],
    'merch_long': [-75.1],
    'gender': ['M'],  # 'M' for Male, 'F' for Female
    'trans_date_trans_time': ['2020-06-21 12:14:25'],
    'merchant': ['fraud_Leffler-Goldner'],
    'category': ['food_dining'],
    'first': ['John'],
    'last': ['Doe'],
    'street': ['508 Erin Mount'],
    'city': ['New York City'],
    'state': ['CA'],
    'job': ['Clinical research associate'],
    'dob': ['2000-01-01'],
    'trans_num': ['0ad27a9cf7fcb1e0774a86709cb248f1']
}

# Convert the dictionary to a DataFrame
custom_input_df = pd.DataFrame(custom_input_data)

# Modify 'gender' column to numeric value (0 for 'M', 1 for 'F')
custom_input_df['gender'] = custom_input_df['gender'].apply(lambda x: 0 if x == 'M' else 1)

# Preprocess the custom input data (TF-IDF and feature concatenation)
X_custom_input_tfidf = []
for col in text_columns:
    try:
        tfidf_data = tfidf_vectorizers[col].transform(custom_input_df[col])
        X_custom_input_tfidf.append(tfidf_data)
    except ValueError as e:
        print(f"Skipping column '{col}' due to error: {e}")

# List of columns to include in X_custom_input_final
included_columns = ['cc_num', 'amt', 'zip', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long', 'gender']
X_custom_input_final = hstack([custom_input_df[included_columns].values.astype('float64')] + X_custom_input_tfidf, format='csr')

# Make predictions on custom input
custom_predictions = model.predict(X_custom_input_final)

# Display the prediction result
if custom_predictions[0] == 0:
    print("\nThe model predicts that the transaction is not fraudulent.")
else:
    print("\nThe model predicts that the transaction is fraudulent.")
