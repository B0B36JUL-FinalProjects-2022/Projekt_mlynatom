using Revise
using Projekt_mlynatom
using DataFrames
using StatsPlots
using Plots
using Flux
using Flux: crossentropy
using Statistics

df = read_csv_to_df("data/train.csv")


# View data
### histograms
@df df histogram(:SibSp, title="Siblings/Spouses")

@df df histogram(:Pclass, title="Passenger classes")

@df df histogram(:Age, title="Age", legend=false)

@df df histogram(:Parch, title="Parch")

@df df histogram(:Fare)

### Dataframe description
describe(df)


# Data augmentation
## Age
#idx = findall(.!completecases(df, :Age))
### fill missing age by mean of ages of other people of same sex and class
fill_missing_age!(df)
describe(df)

## Embarked
count_all(df.Embarked)
### highest occurence has "S"
fill_missing_embarked!(df, "S")

describe(df)


# Prepare training data
dummy_cols = [:Sex, :Pclass, :Embarked]
X_cols = [:Age, :SibSp, :Parch, :Fare]
X, y = prepare_data(df, dummy_cols, X_cols, :Survived)

# Prepare test data
df_test = read_csv_to_df("data/test.csv")

describe(df_test)

##1 missing Fare
findall(.!completecases(df_test, :Fare))
df_test[153, :] #with missing fare -> Embarked: S, Pclass: 3 -> Fare -> mean
using Query

sel_fares_df = @from row in dropmissing(df_test, :Fare) begin
    @where row.Embarked == "S" && row.Pclass == 3
    @select {
        row.Fare,
    }
    @collect DataFrame
end

fare_mean = mean(sel_fares_df.Fare)

df_test[153, :Fare] = fare_mean
describe(df_test)

## missing Age
fill_missing_age!(df_test)
describe(df_test)

##Prepare data
X_test = prepare_data(df_test, dummy_cols, X_cols)



#Logistic regression
X_log = hcat(X, ones(size(X, 1)))

w = log_reg(X_log, y, max_iter=1000)

preds = predict(X_log, w)

error = compute_class_error(y, preds)



# NN
#split dataset
X_train, y_train, X_dev, y_dev = split_dataset(X, y)

#standardize
X_train, X_dev = standardize(X_train', X_dev'; dims=2)

y_train = categorical_to_one_hot(y_train)'
y_dev = categorical_to_one_hot(y_dev)'


my_network = Chain(
    Dense(size(X_train, 1) => 32, relu),
    Dropout(0.9),
    Dense(32 => 32, relu),
    BatchNorm(32),
    Dropout(0.9),
    Dense(32 => size(y_train, 1), identity),
    softmax,
)


loss(X, y) = crossentropy(my_network(X), y)
opt = Adam(0.001)
n_epochs = 1000
acc_test, acc_train, Ls = train_nn!(my_network, loss, X_train, y_train, X_dev, y_dev; opt=opt, n_epochs=n_epochs, batchsize=8)
plot(acc_test, xlabel="Iteration", ylabel="Dev accuracy", label="", ylim=(-0.01, 1.01))
plot!(acc_train, xlabel="Iteration", ylabel="Train accuracy", label="", ylim=(-0.01, 1.01))
plot(Ls)

testmode!(my_network)
accuracy(my_network, X_dev, y_dev; dims=2)
accuracy(my_network, X_train, y_train; dims=2)



#evaluate on test data
## NN
X_nn_test = standardize(X_test; dims=1)

test_preds = predict(X_nn_test', my_network; dims=2)

## log_reg
X_log_test = hcat(X_test, ones(size(X_test, 1)))

test_preds = predict(X_log_test, w)


save_my_submission(test_preds, df_test.PassengerId)