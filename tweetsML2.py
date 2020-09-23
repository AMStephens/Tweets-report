import sqlite3
import pandas as pd
import nltk
import math

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

# Connect to the database
conn = sqlite3.connect('tweets.sqlite')
cur = conn.cursor()

cur.execute('''SELECT just_text, retweet_count, user_id, sentiment, topic 
            FROM tweets 
            WHERE STRFTIME('%Y', tweets.created_at) = '2017'
            ''')

# Grab information into lists
rows = cur.fetchall()
text = [i for i, j, k, l, m in rows]
retweet = [j for i, j, k, l, m in rows]
user = [k for i, j, k, l, m in rows]
sent = [l for i, j, k, l, m in rows]
topic = [m for i, j, k, l, m in rows]

df = pd.DataFrame(zip(text, retweet, user, sent, topic), columns = ['text', 'retweet', 'user', 'sent', 'topic'])
# Convert relevant data to categorical type
df['user'] = df['user'].astype('category').cat.codes
df['user'] = pd.Categorical(df['user'], categories = df['user'].unique())
df['topic'] = df['topic'].astype('category').cat.codes
df['topic'] = pd.Categorical(df['topic'], categories = df['topic'].unique())
# Only include tweets with retweet count of less than 100; limits model scope to improve performance
df = df[df['retweet'] < 100]
df = df.reset_index()
# Limit to 70,000 data instances
df = df.truncate(before = 118924)

# Remove stopwords and any irrlevant abbreviations from the text
new_stopwords = ['rt','amp', '']
stopwords2 = list(ENGLISH_STOP_WORDS)
stopwords2 = set(stopwords2 + new_stopwords)
df['text'] = df['text'].apply(
    lambda word: [word.lower() for word in word.split(' ') if word.lower() not in stopwords2 and word.isalpha() == True]
    )
# Stem the text to combine words with similar meanings
porter = nltk.PorterStemmer()
df['text'] = df['text'].apply(lambda t: [porter.stem(word) for word in t])
df['text'] = df['text'].apply(lambda t: " ".join(t))

X = df[['text', 'user', 'topic', 'sent']]
Y = df['retweet']

# Vectorize text to find most meaningful words and word pairs
vect = TfidfVectorizer(min_df = 25, lowercase = True, ngram_range = (1,2), 
                       stop_words = stopwords2).fit(X['text'])

X_vect = vect.transform(X['text'])

# Reduce high dimensionality 
svd = TruncatedSVD(n_components = 2, random_state = 0)
svd = svd.fit(X_vect)
fitted = svd.transform(X_vect)

def add_feature(X, feature_to_add):
    # Returns sparse feature matrix with added feature
    # Feature_to_add can also be a list of features
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')

X_user = add_feature(fitted, [X['user']])
X_top = add_feature(X_user, [X['topic']])
X_final = add_feature(X_top, [X['sent']])

X_train, X_test, y_train, y_test = train_test_split(X_final, Y, random_state = 0)

# Create model
#model = RandomForestRegressor(n_estimators = 340, max_depth = 26, random_state = 0).fit(X_train, y_train)
#model = SVR(C = 36, gamma = 'auto').fit(X_train, y_train)
model = GradientBoostingRegressor(
    learning_rate = 0.05, n_estimators = 550, max_depth = 6, subsample = 0.8, random_state = 0
    ).fit(X_train, y_train)
#model = ExtraTreesRegressor(n_estimators = 100, random_state = 0).fit(X_train, y_train)

# Evaluate model
pred = model.predict(X_test)
rms = math.sqrt(mean_squared_error(y_test, pred))
print('Root Mean Squared Error: ', rms)

# Create dummy model for comparison
dummy = DummyRegressor(strategy = 'mean').fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
rms = math.sqrt(mean_squared_error(y_test, dummy_pred))
print('Root Mean Squared Error (Dummy): ', rms)

# Optimise parameters - gives the best score and best parameter(s)
#param_grid = {'min_samples_leaf' : [1, 2, 3, 4, 5]}
#grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = 2)
#grid.fit(X_train_final, y_train)
#print(grid.best_score_)
#print(grid.best_estimator_.min_samples_leaf)
