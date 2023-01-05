using CSV
using DataFrames
using Statistics
using Random

export categorical_to_one_hot, categorical_to_dummy_encoding, standardize, split_dataset, one_hot_to_one_cold, fill_missing_age!, fill_missing_embarked!, count_all, prepare_data

function categorical_to_one_hot(vec)
    unique_vals = unique(vec)
    ret_mat = zeros(Bool, (length(vec), length(unique_vals)))

    for i in eachindex(unique_vals)
        ret_mat[vec.==unique_vals[i], i] .= true
    end

    return ret_mat
end

function categorical_to_dummy_encoding(vec)
    unique_vals = unique(vec)
    ret_mat = zeros(Bool, (length(vec), length(unique_vals)))

    for i in eachindex(unique_vals)
        ret_mat[vec.==unique_vals[i], i] .= true
    end

    return ret_mat[:, 2:end]
end


function one_hot_to_one_cold(mat; dims=1)
    ret = [(argmax(vec) - 1) for vec in eachslice(mat; dims=dims)]
    return ret
end

function standardize(X::AbstractMatrix{<:Number}; dims=1)
    col_mean = mean(X; dims=dims)
    col_std = std(X; dims=dims)
    return (X .- col_mean) ./ col_std
end

function standardize(X_train::AbstractMatrix{<:Number}, X_dev::AbstractMatrix{<:Number}; dims=1)
    col_mean = mean(X_train; dims=dims)
    col_std = std(X_train; dims=dims)

    X_train_s = (X_train .- col_mean) ./ col_std
    X_dev_s = (X_dev .- col_mean) ./ col_std

    return X_train_s, X_dev_s
end

function split_dataset(X, y; dev_ratio=0.1)
    n = length(y)
    n_dev = round(Int64, dev_ratio * n)

    idx_rand = randperm(n)
    idx_dev = idx_rand[1:n_dev]
    idx_train = idx_rand[n_dev+1:end]

    X_train = X[idx_train, :]
    X_dev = X[idx_dev, :]
    y_train = y[idx_train]
    y_dev = y[idx_dev]

    return X_train, y_train, X_dev, y_dev
end

function mean_of_age(df, Sex, Pclass)
    class_bit = df[:, :Pclass] .== Pclass
    sex_bit = df[:, :Sex] .== Sex

    return round(mean(df[class_bit.&sex_bit, :Age]))
end

function set_missing_age!(df, Sex, Pclass, age)
    class_bit = df[:, :Pclass] .== Pclass
    sex_bit = df[:, :Sex] .== Sex
    missing_bit = df[:, :Age] .=== missing

    df[class_bit.&sex_bit.&missing_bit, :Age] .= age

    return
end

function fill_missing_age!(df::DataFrame; col_name::Symbol=:Age)
    not_missing_age_df = dropmissing(df, col_name)

    first_men_age = mean_of_age(not_missing_age_df, "male", 1)
    first_women_age = mean_of_age(not_missing_age_df, "female", 1)
    second_men_age = mean_of_age(not_missing_age_df, "male", 2)
    second_women_age = mean_of_age(not_missing_age_df, "female", 2)
    third_men_age = mean_of_age(not_missing_age_df, "male", 3)
    third_women_age = mean_of_age(not_missing_age_df, "female", 3)

    set_missing_age!(df, "male", 1, first_men_age)
    set_missing_age!(df, "female", 1, first_women_age)
    set_missing_age!(df, "male", 2, second_men_age)
    set_missing_age!(df, "female", 2, second_women_age)
    set_missing_age!(df, "male", 3, third_men_age)
    set_missing_age!(df, "female", 3, third_women_age)
    return
end

function fill_missing_embarked!(df::DataFrame, val::String)
    embarked_vec = df.Embarked
    embarked_non_missing = coalesce.(embarked_vec, val)
    df[:, :Embarked] = embarked_non_missing
    return
end

function count_all(vec)
    ret_dict = Dict()
    for val in vec
        if !haskey(ret_dict, Symbol(val))
            ret_dict[Symbol(val)] = 0
        end
        ret_dict[Symbol(val)] += 1
    end
    return ret_dict
end

function prepare_data(df::DataFrame, to_dummy_cols::AbstractVector{Symbol}, X_cols::AbstractVector{Symbol})
    X = Matrix{Float32}(df[:, X_cols])

    for dummy_col in to_dummy_cols
        X = hcat(X, categorical_to_dummy_encoding(df[:, dummy_col]))
    end

    return X
end

function prepare_data(df::DataFrame, to_dummy_cols::AbstractVector{Symbol}, X_cols::AbstractVector{Symbol}, y_col::Symbol)
    X = prepare_data(df, to_dummy_cols, X_cols)
    y = df[:, y_col]

    return X, y
end

