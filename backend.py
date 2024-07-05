import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import os
from sklearn.preprocessing import LabelEncoder

np.random.seed(0)

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

api_key = 'AIzaSyC6OpHIM-Gx_XdP5KAjG9LmvKbvx9kGHIA'
spreadsheet_id = '1LJdBQOxx2cZvu20nVrMnXQ0b8G16X-0_H2-9IytiWgU'
range_name = 'Form%20responses%201'  # URL encoding for the space

url = f'https://docs.google.com/spreadsheets/d/{spreadsheet_id}/gviz/tq?tqx=out:csv&sheet={range_name}'

response = requests.get(url)
if response.status_code == 200:

    df = pd.read_csv(url, index_col=0)
    print(df.head())
else:
    print("Failed to retrieve data.")

columns_to_check = [
    "What is your name",
    "How satisfied are you with the variety of food options available on campus",
    "On a scale of 1 to 5, how would you rate the taste and quality of the food served in the campus cafeteria?",
    "How often do you eat meals on campus? ",
    "Do you have any dietary restrictions or food allergies? If yes, please specify.",
    "Are you satisfied with the affordability of food options on campus?",
    "How would you rate the cleanliness and hygiene of the campus dining facilities?",
    "Do you feel that the campus dining facilities adequately cater to your nutritional needs?",
    "Have you ever experienced any issues with food safety or foodborne illnesses on campus?",
    "Do you have any additional comments or suggestions for improving the campus dining experience?",
    "How much money, on average, do you spend on food per week on campus? (Please enter a numerical value in your local currency)",
    "How many food items do you typically purchase per visit to the campus cafeteria? (Please enter a numerical value)"
]

df['What is your name'].is_unique
df[
    'How much money, on average, do you spend on food per week on campus? (Please enter a numerical value in your local currency)'] = pd.to_numeric(
    df[
        'How much money, on average, do you spend on food per week on campus? (Please enter a numerical value in your local currency)'],
    errors='coerce')
df[
    'How many food items do you typically purchase per visit to the campus cafeteria? (Please enter a numerical value)'] = pd.to_numeric(
    df[
        'How many food items do you typically purchase per visit to the campus cafeteria? (Please enter a numerical value)'],
    errors='coerce')

df.dropna(subset=columns_to_check, axis=0, inplace=True)
numeric_columns = df[columns_to_check].select_dtypes(include=np.number).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

non_numeric_columns = df[columns_to_check].select_dtypes(exclude=np.number).columns
df[non_numeric_columns] = df[non_numeric_columns].ffill(axis=1)
numeric_columns.isna()
non_numeric_columns.isna()
nan_present = df.isnull().values.any()
print("Are there any missing values after preprocessing?", nan_present)

df.info()

df.drop_duplicates(inplace=True)
df[[
    'How much money, on average, do you spend on food per week on campus? (Please enter a numerical value in your local currency)',
    'How many food items do you typically purchase per visit to the campus cafeteria? (Please enter a numerical value)']].describe()
print(df[[
    'How much money, on average, do you spend on food per week on campus? (Please enter a numerical value in your local currency)',
    'How many food items do you typically purchase per visit to the campus cafeteria? (Please enter a numerical value)']].describe())


def detect_outliers_iqr(df):
    outliers = []
    if len(df) == 0:
        return outliers
    df = sorted(df)
    q1 = np.percentile(df, 25)
    q3 = np.percentile(df, 75)
    IQR = q3 - q1
    lower_bound = q1 - (1.5 * IQR)
    upper_bound = q3 + (1.5 * IQR)
    for i in df:
        if i < lower_bound or i > upper_bound:
            outliers.append(i)
    return outliers


combined_data = list(df[
                         'How much money, on average, do you spend on food per week on campus? (Please enter a numerical value in your local currency)']) + list(
    df[
        'How many food items do you typically purchase per visit to the campus cafeteria? (Please enter a numerical value)'])

# Call the function to detect outliers
outliers = detect_outliers_iqr(combined_data)

print("Outliers from IQR method:", outliers)
label_encoder = LabelEncoder()
for i in df.columns:
    if df[i].dtype == 'object':
        df[i] = label_encoder.fit_transform(df[i])
df.head()
corr_matrix = df.corr().abs()
corr_matrix.style.background_gradient(cmap='magma')
dataset_train = df
columns_to_plot = [col for col in dataset_train.columns if
                   col != 'How much money, on average, do you spend on food per week on campus? (Please enter a numerical value in your local currency)']
num_cols = 3
num_rows = int(np.ceil(len(columns_to_plot) / num_cols))
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))
for i, col in enumerate(columns_to_plot):
    row_index = i // num_cols
    col_index = i % num_cols
    sns.regplot(data=dataset_train, x=col,
                y='How much money, on average, do you spend on food per week on campus? (Please enter a numerical value in your local currency)',
                ax=axes[row_index, col_index])
    axes[row_index, col_index].set_title(
        f'{col} vs How much money, on average, do you spend on food per week on campus? (Please enter a numerical value in your local currency)')

num_features = len(columns_to_plot)
for i in range(num_features, num_rows * num_cols):
    axes.flatten()[i].axis('off')
plt.tight_layout()
plt.show()















