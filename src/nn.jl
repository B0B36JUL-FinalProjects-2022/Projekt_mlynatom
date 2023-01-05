using Flux
using Flux: Descent, params, DataLoader
using BSON

export train_nn!, predict, accuracy

"""
    train_nn!(nn, loss, X_train, y_train, X_dev, y_dev; opt=Descent(0.1), n_epochs=30, batchsize=32)

Train loop for neural network `nn` using Flux framework on train data `X_train`, `y_train`,
evaluates precision on development data `X_dev`, `y_dev`. It uses optimizer `opt` for
number of epochs `n_epochs` with batch size `batchsize`.
"""
function train_nn!(nn, loss, X_train, y_train, X_dev, y_dev; opt=Descent(0.1), n_epochs=30, batchsize=32)
    trainmode!(nn) ## according to documentation
    ps = params(nn)

    batches = DataLoader((X_train, y_train); batchsize, shuffle=true)

    acc_test = zeros(n_epochs)
    acc_train = zeros(n_epochs)
    loss_vec = zeros(n_epochs)

    for i in 1:n_epochs
        Flux.train!(loss, ps, batches, opt)
        acc_test[i] = accuracy(nn, X_dev, y_dev; dims=2)
        acc_train[i] = accuracy(nn, X_train, y_train; dims=2)
        loss_vec[i] = loss(X_train, y_train)
    end

    testmode!(nn)
    return acc_test, acc_train, loss_vec
end

"""
    predict(X::AbstractMatrix{<:Number}, my_model::Chain; dims=1)

Predicts values for input matrix `X` given nerual network `my_model`
by dimension `dims`.
"""
function predict(X::AbstractMatrix{<:Number}, my_model::Chain; dims=1)
    return one_hot_to_one_cold(my_model(X); dims=dims)
end

"""
    accuracy(nn::Chain, X, y; dims=1)

Computes accuracy of given neural network `nn` on data matrix `X` and true classes `y`
along dimension `dims`.
"""
function accuracy(nn::Chain, X, y; dims=1)
    return mean(predict(X, nn; dims=dims) .== one_hot_to_one_cold(y; dims=dims))
end