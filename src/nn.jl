using Flux
using Flux: Descent, params, DataLoader
using BSON

export train_nn!, predict, accuracy

function train_nn!(nn, loss, X_train, y_train, X_dev, y_dev; opt = Descent(0.1), n_epochs=30, batchsize = 32)
    trainmode!(nn) ## according to documentation
    ps = params(nn)

    batches = DataLoader((X_train, y_train); batchsize, shuffle = true)

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

function predict(X, my_model::Chain; dims=1)
    return one_hot_to_one_cold(my_model(X); dims=dims)
end

function accuracy(nn, X_test, y_test; dims=1)
    return mean(one_hot_to_one_cold(nn(X_test); dims=dims) .== one_hot_to_one_cold(y_test; dims=dims))
end