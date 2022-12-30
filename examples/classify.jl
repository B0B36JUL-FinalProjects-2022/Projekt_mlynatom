using Revise
using Projekt_mlynatom
using DataFrames

df = read_csv_to_df("data/train.csv")

one_hot_sex = categorical_to_one_hot(df.Sex)

X = Matrix(df[:, [:Pclass, :SibSp, :Parch]])
X = hcat(X, one_hot_sex)

y = df.Survived

#X = standardize(X)

w = log_reg(X, y)

preds = predict(X, w)

error = compute_class_error(y, preds)

### on test

df_test = read_csv_to_df("data/test.csv")
categorical_sex = df_test.Sex
one_hot_sex = categorical_to_one_hot(categorical_sex)

X = Matrix(df_test[:, [:Pclass, :SibSp, :Parch]])
X = hcat(X, one_hot_sex)

preds = predict(X, w)

new_df = DataFrame(PassengerId=df_test.PassengerId, Survived=Int8.(preds))

save_df_to_csv(new_df, "data/my_submission.csv")
