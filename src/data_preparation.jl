using CSV
using DataFrames
using Statistics
using Random

export categorical_to_one_hot, standardize, split_dataset

function categorical_to_one_hot(vec::AbstractVector{<:AbstractString})
    unique_vals = unique(vec)
    ret_mat = zeros(Bool, (length(vec), length(unique_vals)))
    
    for i in eachindex(unique_vals)
        ret_mat[vec.==unique_vals[i], i] .= true
    end

    return ret_mat
end

function standardize(X::Matrix{<:Number}; dims=1)
    col_mean = mean(X; dims=dims)
    col_std = std(X; dims=dims)
    return (X .- col_mean) ./ col_std
end

function standardize(X_train::Matrix{<:Number}, X_dev::Matrix{<:Number}; dims=1)
    col_mean = mean(X_train; dims=dims)
    col_std = std(X_train; dims=dims)

    X_train_s = (X_train .- col_mean) ./ col_std
    X_dev_s = (X_train .- col_mean) ./ col_std

    return X_train_s, X_dev_s
end

function split_dataset(X, y; dev_ratio = 0.1)
    n = length(y)
    n_dev = round(Int64, dev_ratio*n)

    idx_rand = randperm(n)
    idx_dev = idx_rand[1:n_dev] 
    idx_train = idx_rand[n_dev+1:end]

    X_train = X[idx_train, :] 
    X_dev = X[idx_dev, :]
    y_train = y[idx_train]
    y_dev = y[idx_dev]

    return X_train, y_train, X_dev, y_dev
end