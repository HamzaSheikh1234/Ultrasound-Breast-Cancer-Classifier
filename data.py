from imports import *


df = pd.read_csv('/home/hamza/Documents/AI/Breast Cancer Prediction/input/Coimbra_breast_cancer_dataset.csv')

X = df
# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['Classification'], inplace=True)
y = X.Classification              
X.drop(['Classification'], axis=1, inplace=True)

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)
train_data = X_train.join(y_train)
#sns.heatmap(train_data.corr(),annot = True, cmap = 'YlGnBu')

# Select numeric columns and get logarithms for more accurate data
# numeric_cols = X_train.columns
# for col in numeric_cols:
#     train_data[col] = np.log(train_data[col]+1)
#     X_valid[col] = np.log(X_valid[col]+1)
# y_valid = np.log(y_valid+1)
#print(train_data.hist(figsize=(15,8)))

y_train = train_data.Classification              
train_data.drop(['Classification'], axis=1, inplace=True)
print(X_valid.head())