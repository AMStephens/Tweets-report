import sqlite3
import pandas as pd
import numpy as np
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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
# Get labels
df['retweet'] = ['a' if x == 0 else 'b' if x > 0 and x <= 100 else 'c' for x in df['retweet']]
df['retweet'] = df['retweet'].astype('category').cat.codes
df['retweet'] = pd.Categorical(df['retweet'], categories = df['retweet'].unique())
# Ensure balanced classes
df0 = df[df['retweet'] == 0]
df0 = df0.reset_index()
df1 = df[df['retweet'] == 1]
df1 = df1.reset_index()
df1 = df1.truncate(after = 17048)
df2 = df[df['retweet'] == 2]
df2 = df2.reset_index()
df2 = df2.truncate(after = 17048)
result = pd.concat([df0, df1, df2])
result = result.reset_index()

new_stopwords = ['rt','amp', '']
stopwords2 = list(ENGLISH_STOP_WORDS)
stopwords2 = set(stopwords2 + new_stopwords)

# Stem the text to combine words with similar meanings
result['text'] = result['text'].apply(
    lambda word: [word.lower() for word in word.split(' ') if word.lower() not in stopwords2 and word.isalpha() == True]
    )
porter=nltk.PorterStemmer()
result['text'] = result['text'].apply(lambda t: [porter.stem(word) for word in t])
result['text'] = result['text'].apply(lambda t: " ".join(t))

X = result[['text', 'user', 'topic', 'sent']]
Y = result['retweet']

# Vectorize text to find most meaningful words and word pairs
vect=TfidfVectorizer(min_df = 25, lowercase = True, ngram_range = (1,2), 
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

n_iterations = 100

# Iterate through multiple train/test splits
# Can set either the train/test split or model random state
stats = list()
for i in range(n_iterations):
    X_train, X_test, y_train, y_test = train_test_split(X_final, Y, random_state=0)
    # Fit model
    model = GradientBoostingClassifier(
        learning_rate = 0.05, n_estimators = 550, max_depth = 6, subsample = 0.8
        ).fit(X_train, y_train)
    pred = model.predict(X_test)
	# Evaluate model
    score = accuracy_score(y_test, pred)
    stats.append(score)

print(stats)

# Plot scores
plt.hist(stats, color='seagreen')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.show()
# Calculate confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(stats, p))
p = (alpha + ((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(stats, p))
print('Average Accuracy Score: ', np.average(stats))
print('%.1f Confidence Interval %.1f%% and %.1f%%' % (alpha * 100, lower * 100, upper * 100))

# Create dummy model for comparison
dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
print('Accuracy Score (Dummy): ', accuracy_score(y_test, dummy_pred))
