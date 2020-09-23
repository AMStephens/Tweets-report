import sqlite3
import pandas as pd
import nltk
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn import model_selection

# Connect to database
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
df['user'] = pd.Categorical(df['user'], categories=df['user'].unique())
df['topic'] = df['topic'].astype('category').cat.codes
df['topic'] = pd.Categorical(df['topic'], categories=df['topic'].unique())
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

# Remove stopwords and any irrlevant abbreviations from the text
new_stopwords = ['rt','amp', '']
stopwords2 = list(ENGLISH_STOP_WORDS)
stopwords2 = set(stopwords2 + new_stopwords)
result['text'] = result['text'].apply(
    lambda word: [word.lower() for word in word.split(' ') if word.lower() not in stopwords2 and word.isalpha() == True]
    )
# Stem the text to combine words with similar meanings
porter = nltk.PorterStemmer()
result['text'] = result['text'].apply(lambda t: [porter.stem(word) for word in t])
result['text'] = result['text'].apply(lambda t: " ".join(t))

X = result[['text', 'user', 'topic', 'sent']]
Y = result['retweet']

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

seed = 0
# Prepare models
models = []
models.append(('SVC', SVC(C = 48, kernel = 'rbf', gamma = 'auto', random_state = 0)))
models.append(('DTC', DecisionTreeClassifier(max_depth = 42, random_state = 0)))
models.append(('RFC', RandomForestClassifier(n_estimators = 250, max_depth = 42, random_state = 0)))
models.append(('SGDC',  SGDClassifier(random_state = 0, n_jobs = -1)))
models.append(('GBC', GradientBoostingClassifier(
    learning_rate = 0.05, n_estimators = 550, max_depth = 6, subsample = 0.8, random_state = 0
    )))
models.append(('DUM', DummyClassifier(strategy = 'most_frequent')))
# Evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits = 15, random_state = seed)
	cv_results = model_selection.cross_val_score(model, X_final, Y, cv = kfold, scoring = scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.mean())
	print(msg)
# Boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
