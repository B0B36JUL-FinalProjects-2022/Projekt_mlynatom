using Revise
using Projekt_mlynatom
using DataFrames
using StatsPlots

df = read_csv_to_df("data/train.csv")

### View data
#histogram of siblings
@df df histogram(:SibSp, title="Siblings/Spouses")

#histogram of Passenger classes
@df df histogram(:Pclass, title="Passenger classes")

@df df histogram(:Age, title="Age", legend=false)

@df df histogram(:Parch, title="Parch")

@df df histogram(:Fare)

### Prepare data
one_hot_sex = categorical_to_one_hot(df.Sex)

X = Matrix(df[:, [:Pclass, :SibSp, :Parch]])
X = hcat(X, one_hot_sex)

y = df.Survived

### Logistic regression
#X = standardize(X)

w = log_reg(X, y)

preds = predict(X, w)

error = compute_class_error(y, preds)

## on test

df_test = read_csv_to_df("data/test.csv")
categorical_sex = df_test.Sex
one_hot_sex = categorical_to_one_hot(categorical_sex)

X = Matrix(df_test[:, [:Pclass, :SibSp, :Parch]])
X = hcat(X, one_hot_sex)

preds = predict(X, w)

new_df = DataFrame(PassengerId=df_test.PassengerId, Survived=Int8.(preds))

save_df_to_csv(new_df, "data/my_submission.csv")

### NN
X_train, y_train, X_dev, y_dev = split_dataset(X, y)

#standardize
X_train_s, X_dev_s = standardize(X_train, X_dev)

my_network = Chain(
    Dense(size())
)