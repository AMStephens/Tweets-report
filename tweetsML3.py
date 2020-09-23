import sqlite3
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Connect to the database
conn = sqlite3.connect('tweets.sqlite')
cur = conn.cursor()

cur.execute('''SELECT just_text, retweet_count, user_id, sentiment, topic 
            FROM tweets 
            WHERE STRFTIME('%Y', tweets.created_at) IN ('2017', '2016')
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
df['retweet'] = ['a' if x == 0 else 'b' for x in df['retweet']]
df['retweet'] = df['retweet'].astype('category').cat.codes
df['retweet'] = pd.Categorical(df['retweet'], categories = df['retweet'].unique())
# Ensure balanced classes
df0 = df[df['retweet'] == 0]
df0 = df0.reset_index()
df0 = df0.truncate(after = 24999)
df1 = df[df['retweet'] == 1]
df1 = df1.reset_index()
df1 = df1.truncate(after = 24999)
result = pd.concat([df0, df1])
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

X_train, X_test, y_train, y_test = train_test_split(X_final, Y, random_state = 0)

# Create model
#model = SVC(C = 50, kernel = 'rbf', gamma = 'auto', random_state = 0).fit(X_train, y_train)
#model = DecisionTreeClassifier().fit(X_train, y_train)
#model = RandomForestClassifier(n_estimators = 160, max_depth = 36, random_state = 0).fit(X_train, y_train)
#model = SGDClassifier(random_state = 0, n_jobs = -1).fit(X_train, y_train)
model = GradientBoostingClassifier(
    learning_rate = 0.05, n_estimators = 550, max_depth = 6, subsample = 0.8, random_state = 0
    ).fit(X_train, y_train)
# Evaluate model
pred = model.predict(X_test)
print(classification_report(y_test, pred))
print('Accuracy Score: ', accuracy_score(y_test, pred))
print('ROC Score: ', roc_auc_score(y_test, pred))

# Create dummy model for comparison
dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
print(classification_report(y_test, dummy_pred))
print('Accuracy Score (Dummy): ', accuracy_score(y_test, dummy_pred))
print('ROC Score (Dummy): ', roc_auc_score(y_test, dummy_pred))

# Optimise parameters - gives the best score and best parameter(s)
#param_grid = {'C' : [63, 70, 80]}
#grid = GridSearchCV(estimator = model, scoring = 'accuracy', param_grid = param_grid, cv=2)
#grid.fit(X_train_final, y_train)
#print((grid.best_score_))
#print(grid.best_estimator_.C)

# Create confusion matrix based on results
conf_mat = confusion_matrix(y_test, pred)

class_names = ['no retweets', 'some retweets']
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(conf_mat), annot = True, cmap = 'YlGnBu', fmt = 'g', cbar = True)
ax.xaxis.set_label_position('top')
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)
plt.title('Confusion Matrix', y = 1.1)
plt.ylabel('Actual Label')
plt.xlabel('Predicted label')
