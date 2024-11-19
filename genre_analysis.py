import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import zipfile
import warnings
import pathlib
warnings.filterwarnings('ignore')

# Run on google colab or locally or on Kaggle 

if os.path.exists('/kaggle/input'):
    print('Running on Kaggle...')
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
            # Read the CSV file from Kaggle
            df = pd.read_csv(os.path.join(dirname, filename))
    print("Data loaded successfully from Kaggle!")
else:
    try:
        from google.colab import drive
        print('Running on CoLab...')
        drive.mount('/content/drive')
        # Read the csv file from Collab
        df = pd.read_csv('/content/drive/My Drive/ClassicHit.csv')
        print("Done!")
    except:
        print("Not running on Kaggle")
        # Download the data
        # This is the link for the data set
        # https://www.kaggle.com/datasets/thebumpkin/10400-classic-hits-10-genres-1923-to-2023/data
        !kaggle datasets download -d thebumpkin/10400-classic-hits-10-genres-1923-to-2023
    
        # Jupyter Notebook's path
        desktop = pathlib.Path.home() / 'Desktop'
        os.chdir(desktop) # This automatically transfer you to Desktop (suppose that your jupyter notebook is on Desktop)
    
        # Unzip the files
        with zipfile.ZipFile('./10400-classic-hits-10-genres-1923-to-2023.zip', 'r') as zip_ref:
            zip_ref.extractall('./Genre_Dataset')
    
    
        print('Changing Directory!')
        os.chdir('./Genre_Dataset')
        print('Current Directory: ', os.getcwd())
    
        # Read the csv from Desktop locally
        df = pd.read_csv('ClassicHit_cleaned.csv')
    
        print("Done!")

  '''
About Dataset

The dataset is a comprehensive collection of 15,150 classic hits from 3,083 artists, spanning
a century of music history from 1923 to 2023. This diverse dataset is divided into 19 distinct genres,
showcasing the evolution of popular music across different eras and styles. Each track in the dataset is
enriched with Spotify audio features, offering detailed insights into the acoustic properties, rhythm, tempo,
and other musical characteristics. This makes the dataset not only a valuable resource for exploring trends and
comparing genres but also for analyzing the sonic qualities that define classic hits across different time periods and genres.
'''

# Read the csv file
display(df)

# Print the head of the data set
df.head()

# print the shape aka the number of rows, columns
print(f"The number of rows are: {df.shape[0]} and the number of columns are: {df.shape[1]}")

# print the columns and the type of each column
df.info()

# print the descriptive statistics
df.describe()

# Check if there are any missing values
print(f"The total missing values are: {df.isnull().sum().sum()}")

# Check the number of artists and also the number of genres and which genres appeared in the data set
print(f"The number of artists are {df['Artist'].unique().shape[0]}")
print(f"The number of genres are {df['Genre'].unique().shape[0]}")
print(f"The Genres are: {', '.join(df['Genre'].unique())}")

# As we can see from the data set, the columns Time_Signature, Key and Mode are categorical attributes. So we'll find the number of unique values
print(f"Time_Signature column unique values: {df['Time_Signature'].unique().shape[0]}")
print(f"Key column unique values: {df['Key'].unique().shape[0]}")
print(f"Mode column unique values: {df['Mode'].unique().shape[0]}")

# First we are going to change the duration to seconds because we suppose that the duration column is milliseconds
df_v1 = df.copy()
df_v1['Duration'] = round(df_v1['Duration']/1000).astype(int)

# 1. We'll start with the column 'Genre'

plt.figure(figsize=(12, 4))
sns.barplot(df.groupby('Genre').count()['Track'].reset_index(), x = 'Genre', y = 'Track')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# 2. Due to the fact that the columns 'Year', 'Duration', Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness',
# 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Popularity' are numerical variables/attributes we are going to use
# for each attribute a histogram plot to observe how the values are distributed
# So we are going to create 12 histogram plots all together (6x2)


def histogram_plot(col1, col2):
    f, axs = plt.subplots(1, 2, figsize=(12, 6))
    sns.histplot(data = df, x = col1, ax = axs[0])
    axs[0].set_title(col1)
    sns.histplot(data = df, x = col2, ax = axs[1])
    axs[1].set_title(col2)
    plt.show()

# Function to send two elements at a time
def send_columns(arr):
    for i in range(0, len(arr), 2):
        if i + 1 < len(arr):
            histogram_plot(arr[i], arr[i + 1])

columns = ['Year', 'Duration', 'Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness','Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Popularity']
send_columns(columns)

# 3. For the columns 'Time_Signature', 'Key' and 'Mode' we are going to plot a bar plot

def bar_plot(col):
    plt.figure(figsize=(12, 4))
    fun_data = pd.DataFrame(df[col].value_counts()).reset_index()
    fun_data = fun_data.rename(columns={'count': 'Count'})
    sns.barplot(data = fun_data, x = col, y = 'Count')
    plt.title(col)
    plt.show()

columns = ['Time_Signature', 'Key', 'Mode']
for column in columns:
    bar_plot(column)

# Now that we are done with each column individually, we are going to compine two attributes, e.g scatterplots
# 4. We'll use scatterplot to see if there is any correlation between Danceability and Energy (we should see positive correlation)
# We map the column 'Mode' to a channel as hue
# We see nothing because of the large amount of data :)
# Maybe we could try to find the correlation stay tuned :)

f, axs = plt.subplots(1, 2, figsize=(12, 6))

sns.scatterplot(data = df, x='Danceability', y = 'Energy', hue = 'Mode', ax = axs[0])
axs[0].set_title('Before')

# First we group by the artist and then we find the average. maybe in that way we will reduce the data and observe any correlation
# We remove the columns Track and Genre because they are string columns
grouped_artists = df.drop(columns=['Track', 'Genre', 'Year'], axis = 1)
grouped_artists = grouped_artists.groupby('Artist').mean().reset_index(drop=True)

sns.scatterplot(data = grouped_artists, x='Danceability', y = 'Energy', ax = axs[1])
axs[1].set_title('After')
plt.show()

# 5. Find the appropriate plot to visualize the correlations using the grouped_artists data frame (Spoiler alert -> Heatmap)
correlations = grouped_artists.corr()
f, axes = plt.subplots(figsize=(14, 6))
sns.heatmap(correlations, annot=True, vmin=-1, vmax=1, linecolor='white', linewidth= 1, cmap="coolwarm",  annot_kws={"size": 10})
axes.set_title('Correlation Heatmap')
plt.show()

# 6. From the map above we can observe high negative correlation between Acousticness and (Energy and Loudness)
# Let's create the two scatterplots to observe this correlation along with a linear regression line
f, axs = plt.subplots(1, 2, figsize=(12, 6))

sns.regplot(data = df, x='Acousticness', y = 'Energy', ax = axs[0], line_kws={"color": "C1"})
axs[0].set_title('Negative Correlation Scatterplot')

sns.regplot(data = df, x='Acousticness', y = 'Loudness', ax = axs[1], line_kws={"color": "C1"})
axs[1].set_title('Negative Correlation Scatterplot')

plt.show()

# 7. For each Genre, plot the average Loudness from highest to lowest value
# First we remove the object columns (Track and Artist)
ave_loud = df.drop(['Track', "Artist"], axis = 1)
ave_loud = ave_loud.groupby('Genre').mean().reset_index().loc[:,['Genre', "Loudness"]]
ave_loud['Loudness'] = -ave_loud['Loudness']
plt.subplots(figsize=(14, 6))
sns.barplot(data = ave_loud.sort_values(by = 'Loudness', ascending=False).reset_index(drop=True), x= 'Genre', y = 'Loudness')
plt.xticks(rotation=45)
plt.ylabel('Loudness (-dB)')
plt.show()

# 1. Which track has the highest popularity?
# Query: Find the track with the maximum popularity value.

most_popular = df.loc[df['Popularity'].idxmax(),'Track']
print(f"The most pupolar song is: {most_popular}")

df.loc[df['Popularity'].idxmax(),'Track']

# 2. What is the average danceability of tracks from 2020?
# Query: Calculate the mean danceability score for tracks released in the year 2020

mean_track_2020 = df.loc[df['Year'] == 2020, 'Danceability'].mean()
print(f"The average danceability of tracks from 2020 is: {round(mean_track_2020, 3)}")

# 3. What is the most common time signature in the dataset?
# Query: Determine the mode (most frequent) value of the "Time_Signature" column.

most_frequent_time_signature = df['Time_Signature'].value_counts().idxmax()

print(f"The most common time signature in the dataset is: {most_frequent_time_signature} with number of value: {df['Time_Signature'].value_counts().max()}")

# 4. How many unique genres are there in the dataset?
# Query: Count the number of distinct genres present in the "Genre" column.
# (This was answered above in the bar plot about the Genre column)
df['Genre'].value_counts()

# 5. What is the average energy level of the top 10% most popular tracks?
# Query: Filter the top 10% most popular tracks and compute the average "Energy" for those tracks.

top_ten_percent = df.shape[0]//10
most_popular_tracks_energy = df.sort_values(by='Popularity', ascending = False).reset_index(drop=True).loc[0:top_ten_percent,'Energy'].mean()
print(f"The average energy level of the top 10% most popular tracks is: {round(most_popular_tracks_energy,3)}")

# 6. Which genre has the highest average acousticness?
# Query: Group the dataset by "Genre" and calculate the mean acousticness, then find which genre has the highest value.

highest_average_acousticness = df.drop(['Track', 'Artist'], axis = 1)
highest_average_acousticness = highest_average_acousticness.groupby('Genre').mean()
highest_average_acousticness = highest_average_acousticness['Acousticness']

print(f"The Genre with the highest average acousticness of {round(highest_average_acousticness.max(), 3)} is: {highest_average_acousticness.idxmax()}")

# 7. How many tracks have a speechiness greater than 0.05 and are from the year 2015?
# Query: Filter tracks with speechiness > 0.5 and released in 2015, then count how many such tracks exist.

num_tracks_Q7 = df.loc[(df['Speechiness'] > 0.05) & (df['Year'] == 2015),'Track'].count()
print(f"The number of tracks that have a speechiness greater than 0.05 and are from the year 2015 is: {num_tracks_Q7}")

# 8. What is the average tempo for tracks with instrumentalness equal to 0?
# Query: Calculate the mean tempo of tracks where the "Instrumentalness" is equal to 0.

mean_tempo_tracks = df.loc[(df['Instrumentalness'] == 0), 'Tempo']
mean_tempo_tracks = mean_tempo_tracks.mean()
print(f"The average tempo for tracks with instrumentalness equal to 0 is {round(mean_tempo_tracks, 3)}")

# 9. Which artist has released the most tracks in the dataset, and what is the average popularity of their tracks?
# Query: Find the artist with the highest number of tracks, and compute the average popularity of their tracks.

popular_artist = df['Artist'].value_counts().reset_index().iloc[0,0]
print(f"The artist who has released the most tracks in the dataset, is: {popular_artist}")

average_popularity = df.loc[df['Artist'] == popular_artist, "Popularity"].mean()
print(f"The average popularity of {popular_artist}, is: {round(average_popularity, 2)}")

# 10. How has the average loudness of tracks changed over the years?
# Query: Group the data by "Year" and calculate the average loudness for each year, then plot the trend over time.

yearly_average_loudness = df.drop(['Track', 'Artist', 'Genre'], axis = 1)
yearly_average_loudness = yearly_average_loudness.groupby('Year').mean().reset_index()[['Year', "Loudness"]]

sns.lineplot(data = yearly_average_loudness, x = "Year", y = "Loudness")
plt.title("Average loudness for each year")
plt.ylabel("Loudness (dB)")
plt.show()

# 11. What is the relationship between danceability and energy across different genres?
# Query: Group the data by "Genre" and calculate the correlation between "Danceability" and "Energy" within each genre.

rel = df.drop(['Artist', 'Track'], axis = 1)
rel = rel.groupby('Genre').corr()[['Danceability', 'Energy']].reset_index()
rel = rel.loc[(rel['level_1'] == "Danceability") | (rel['level_1'] == "Energy"), :]
rel = rel.set_index("level_1")
rel

# 12. Which key is most associated with high valence tracks (valence > 0.8)?
# Query: Filter for tracks with a valence greater than 0.8, then find the key that appears most frequently in this subset.
filtered_tracks = df[df.Valence > 0.8]
filtered_tracks_key = filtered_tracks["Key"].value_counts().reset_index()
print(f"The most associated key with valence > 0.8 is: {filtered_tracks_key.iloc[0,0]} and the number of tracks are: {filtered_tracks_key.iloc[0,1]}")

# 13. How has the average danceability, energy, and valence of tracks changed across different decades?
# First, group the dataset by decades (e.g., 1960s, 1970s, etc.), based on the "Year" column. (it's better to create a new variable called Decade)
# Then, calculate the average "Danceability," "Energy," and "Valence" for each decade.
# Finally, visualize the changes for each feature across decades using a multi-line plot and three boxplots of each column

unique_years = np.sort(df['Year'].unique())
print(f"The unique years are: \n{unique_years}")
# Here we have the year 1899 which technically belongs to 1890s but we have only one value so we put it in the decade 1900s
decade = [year+1 if year == 1899  else year - (year%10) for year in df['Year']]
# Add the list decade to the data frame
df_new = df.copy()
df_new['Decade'] = decade

average_dec = df_new.drop(['Track', 'Artist', 'Genre'], axis = 1)
average_dec = average_dec.groupby('Decade').mean()[['Danceability', 'Energy', 'Valence']].reset_index()

# We use this melt function in order to create the lineplot with multiple lines
average_dec = pd.melt(average_dec, ['Decade']).rename(columns = {"variable": "Attributes", "value": "Attribute Values"})
# print(average_dec) -> uncomment to see the output

# This is the lineplot
plt.subplots(figsize=(8, 6))
sns.lineplot(data = average_dec, x = "Decade", y='Attribute Values', hue='Attributes')
plt.title("Danceability/Energy/Valence over the Decades")
plt.show()

# We finalize this chapter with the boxplots for each Decade
decade_values = df_new.drop(['Track', 'Artist', 'Genre'], axis = 1)[['Decade', 'Danceability', 'Energy', 'Valence']]

def boxplots(col):
    if col == "Decade":
        pass
    else:
        plt.subplots(figsize=(14, 7))
        sns.boxplot(data=decade_values, x="Decade", y=col)
        plt.title("BoxPlot of "+col)
        plt.show()

for col_name in decade_values.columns:
    boxplots(col_name)

# Classification -> Random Forest and kNN
# Regression -> LS, Ridge and Lasso Regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble  import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Additional classification; Recurrent Neural Network (RNN) model
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

# Add in RNN implmentation here

# Classification
# First we are going to split the data into training and testing
X = df.drop(['Track', 'Artist', 'Year', 'Genre'], axis = 1) # Remove Artist, Track and Year because they don't provide any usefull information about the Genre
y = df['Genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)

if set(y_train) == set(y_test):
    print("The unique elements of the training and testing class variable Genre are the same.")
else:
    print("Check again the train_test_split() function!")

# This function is responsible for print the evaluation
def result_classification(test,pred):
    print(print(f"The accuracy score is: {round(accuracy_score(test, pred), 3)} \n"))
    class_report = classification_report(test, pred, target_names = test.unique())
    print(class_report)

# KNN Algorithm
knn_model = KNeighborsClassifier(n_neighbors = 7)
knn_fit = knn_model.fit(X_train, y_train)
knn_test_preds = knn_fit.predict(X_test)
result_classification(y_test, knn_test_preds)

# Random Forest Algorithm
rf_model = RandomForestClassifier(n_estimators = 500, criterion = 'gini', max_depth = 4)
rf_fit = rf_model.fit(X_train, y_train)
rf_test_preds = rf_fit.predict(X_test)
result_classification(y_test, rf_test_preds)

# Regression LS, Ridge and Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# Prepare the dataset
# Our y variable will the column Popularity and will use some of the columns

y = df['Popularity']
X = df.drop(['Year', 'Artist', 'Genre', 'Track', 'Popularity'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=12345)

def regression_results(model_fit, test, pred):
    print(f"The coefficients are: \n {[round(coef ,4) for coef in model_fit.coef_]}")
    print(f"The intercept is: \n {model_fit.intercept_}")

    MSE = np.sum((test - pred)**2)/test.shape[0]
    RMSE = np.sqrt(MSE)
    MAE = np.sum(np.absolute(test - pred))/test.shape[0]
    print(f"The Mean Squared Error is: {round(MSE, 2)}")
    print(f"The Root Mean Squared Error is: {round(RMSE, 2)}")
    print(f"The Mean Absolute Error is: {round(MAE, 2)}")

# LS Regression
LS_model = LinearRegression()
LS_fit = LS_model.fit(X_train, y_train)
LS_pred = LS_fit.predict(X_test)
regression_results(LS_fit, y_test, LS_pred)

# Ridge Regression
Ridge_model = Ridge(alpha=5)
Ridge_fit = Ridge_model.fit(X_train, y_train)
Ridge_pred = Ridge_fit.predict(X_test)
regression_results(Ridge_fit, y_test,Ridge_pred)

# Lasso Regression
Lasso_model = Lasso(alpha=5)
Lasso_fit = Lasso_model.fit(X_train, y_train)
Lasso_pred = Lasso_fit.predict(X_test)
regression_results(Lasso_fit, y_test,Lasso_pred)

# The columns are
class_variable = df['Genre'].unique()
string_classes = ", ".join(class_variable)
string_classes

# 1. We will remove the columns Track, Artist, Year, Duration and Popularity
df_processed = df.drop(['Track', 'Artist', 'Year', 'Duration', 'Popularity'], axis=1)

# 2. Group by Genres and take average of the columns below
df_numeric_columns = df_processed.groupby("Genre").mean()[['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']]

# 3. Remove the columns Key, Mode and Time_Signature because in case you uncomment the below, there is no difference between the each Genre
'''
print(df_processed[['Key', 'Genre']].groupby('Genre').value_counts().reset_index().groupby('Genre').max())
print(df_processed[['Mode', 'Genre']].groupby('Genre').value_counts().reset_index().groupby('Genre').max())
print(df_processed[['Time_Signature', 'Genre']].groupby('Genre').value_counts().reset_index().groupby('Genre').max())
'''

# 4. Final data set
final_df = df_numeric_columns.copy()

# Now we will calculate the cosine similarity between each Genre in order then to group the Genres into large sets
from sklearn.metrics.pairwise import cosine_similarity

cosine = pd.DataFrame(cosine_similarity(final_df))

genre_index = final_df.reset_index()['Genre'] 

cosine.columns = genre_index
cosine.index = genre_index

# If we print the result we could observe than the cosime similarity between all the Genres are close to 1. So we are unable to proceed in order to reduce the Genres to larger sets
display(cosine)
