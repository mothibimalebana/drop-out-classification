import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("student_dropout_dataset_v3.csv", index_col=0)
X_full = df.drop(['Dropout'], axis=1)
y = df['Dropout']


X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, test_size=0.20, train_size=0.80, random_state=0)

categorical_cols = [cname for cname in X_full.columns if X_full[cname].nunique() < 10 and X_full[cname].dtype == "string"]
numerical_cols = [cname for cname in X_full.columns if X_full[cname].dtype in ['int64', 'float64']]
total_cols = categorical_cols + numerical_cols

X_train = X_train_full[total_cols].copy()
X_valid = X_valid_full[total_cols].copy()

numerical_transformer = SimpleImputer(strategy='mean')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
])

X_after = preprocessor.fit_transform(X_train)

feature_names = preprocessor.get_feature_names_out()

X_after_df = pd.DataFrame(X_after, columns=feature_names)

for col in feature_names:
    print(f"column: {col} \tdata type: {X_after_df[col].dtype}")

X_after_df.to_csv('after_cleaning.csv', index=False)

























