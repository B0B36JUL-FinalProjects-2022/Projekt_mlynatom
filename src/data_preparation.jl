using CSV
using DataFrames
using Statistics
using Random
using Query

export categorical_to_one_hot, categorical_to_dummy_encoding, standardize, split_dataset, one_hot_to_one_cold, fill_missing_age!, fill_missing_embarked!, count_all, prepare_data, add_titles!, compute_fare_mean

"""
    categorical_to_one_hot(vec::AbstractVector)

Converts vector `vec` of categorical values to one hot representation.

```julia-repl
julia> categorical_to_one_hot(["a", "b"])
2×2 Matrix{Bool}:
 1  0
 0  1
```
"""
function categorical_to_one_hot(vec::AbstractVector)
    unique_vals = unique(vec)
    ret_mat = zeros(Bool, (length(vec), length(unique_vals)))

    for i in eachindex(unique_vals)
        ret_mat[vec.==unique_vals[i], i] .= true
    end

    return ret_mat
end

"""
    categorical_to_dummy_encoding(vec::AbstractVector)

Converts vector `vec` of categorical values to dummy encoding representation.

```julia-repl
julia> categorical_to_dummy_encoding(["a", "b"])
2×1 Matrix{Bool}:
 0
 1
```
"""
function categorical_to_dummy_encoding(vec::AbstractVector)
    unique_vals = unique(vec)
    ret_mat = zeros(Bool, (length(vec), length(unique_vals)))

    for i in eachindex(unique_vals)
        ret_mat[vec.==unique_vals[i], i] .= true
    end

    return ret_mat[:, 2:end]
end

"""
    one_hot_to_one_cold(mat; dims=1)

Converts matrix `mat` in one hot representation to one cold representation.

```julia-repl
julia> one_hot_to_one_cold([1 0 0; 0 1 0; 0 0 1; 0 0 1])
4-element Vector{Int64}:
 0
 1
 2
 2
```
"""
function one_hot_to_one_cold(mat; dims=1)
    ret = [(argmax(vec) - 1) for vec in eachslice(mat; dims=dims)]
    return ret
end

"""
    standardize(X::AbstractMatrix{<:Number}; dims=1)

Standardize matrix `X` alond dimension `dims`. (set mean to 0 and standard deviation to 1).
"""
function standardize(X::AbstractMatrix{<:Number}; dims=1)
    col_mean = mean(X; dims=dims)
    col_std = std(X; dims=dims)
    return (X .- col_mean) ./ col_std
end

"""
    standardize(X_train::AbstractMatrix{<:Number}, X_dev::AbstractMatrix{<:Number}; dims=1)

Standardize matrices `X_train`, `X_dev` along dimension `dims`. 
(set mean to 0 and standard deviation to 1).
"""
function standardize(X_train::AbstractMatrix{<:Number}, X_dev::AbstractMatrix{<:Number}; dims=1)
    col_mean = mean(X_train; dims=dims)
    col_std = std(X_train; dims=dims)

    X_train_s = (X_train .- col_mean) ./ col_std
    X_dev_s = (X_dev .- col_mean) ./ col_std

    return X_train_s, X_dev_s
end

"""
    split_dataset(X, y; dev_ratio=0.1)

Splits matrix `X` and vector `y` with ratio of development part `dev_ratio`.
"""
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

"""
    mean_of_age(df, Sex, Pclass)

Computes mean of age in certain sex `Sex` and class `Pclass` group in DataFrame
`df`.
"""
function mean_of_age(df, Sex, Pclass)
    class_bit = df[:, :Pclass] .== Pclass
    sex_bit = df[:, :Sex] .== Sex

    return round(mean(df[class_bit.&sex_bit, :Age]))
end

"""
    set_missing_age!(df, Sex, Pclass, age)

Sets given `age` of to missing rows of groups of same `Sex` and `Pclass` in
DataFrame `df`.
"""
function set_missing_age!(df, Sex, Pclass, age)
    class_bit = df[:, :Pclass] .== Pclass
    sex_bit = df[:, :Sex] .== Sex
    missing_bit = df[:, :Age] .=== missing

    df[class_bit.&sex_bit.&missing_bit, :Age] .= age

    return
end

"""
    fill_missing_age!(df::DataFrame; col_name::Symbol=:Age)

Fills missing ages in column of name `col_name` in DataFrame `df`.
"""
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

"""
    fill_missing_embarked!(df::DataFrame, val::String)

Fills missing Embarked rows by values `val` in DataFrame `df`.
"""
function fill_missing_embarked!(df::DataFrame, val::String)
    embarked_vec = df.Embarked
    embarked_non_missing = coalesce.(embarked_vec, val)
    df[:, :Embarked] = embarked_non_missing
    return
end

"""
    count_all(vec)

Counts occurences of elements in vector `vec`. Returns dictionary with
Symbols as keys.

```julia-repl
julia> count_all(["a", "bb", "bb", "c"])
Dict{Any, Any} with 3 entries:
  :a  => 1
  :bb => 2
  :c  => 1

julia> count_all([1, 2, 2, 3, 3, 3])
Dict{Any, Any} with 3 entries:
  Symbol("1") => 1
  Symbol("2") => 2
  Symbol("3") => 3
```
"""
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

"""
    prepare_data(df::DataFrame, to_dummy_cols::AbstractVector{Symbol}, X_cols::AbstractVector{Symbol})

Prepares data in DataFrame `df` to matrix `X`. Columns `to_dummy_cols` of `df`
are converted to dummy encoding, columns `X_cols` are used without changes.
Columns names that are not given are ignored.
"""
function prepare_data(df::DataFrame, to_dummy_cols::AbstractVector{Symbol}, X_cols::AbstractVector{Symbol})
    X = Matrix{Float32}(df[:, X_cols])

    for dummy_col in to_dummy_cols
        X = hcat(X, categorical_to_dummy_encoding(df[:, dummy_col]))
    end

    return X
end

"""
    prepare_data(df::DataFrame, to_dummy_cols::AbstractVector{Symbol}, X_cols::AbstractVector{Symbol}, y_col::Symbol)

Prepares data in DataFrame `df` to matrix `X`. Columns `to_dummy_cols` of `df`
are converted to dummy encoding, columns `X_cols` are used without changes.
Column `y_col` is used for `y`. Columns names that are not given are ignored.
"""
function prepare_data(df::DataFrame, to_dummy_cols::AbstractVector{Symbol}, X_cols::AbstractVector{Symbol}, y_col::Symbol)
    X = prepare_data(df, to_dummy_cols, X_cols)
    y = df[:, y_col]

    return X, y
end

"""
    add_titles!(df::DataFrame, least_occuring_titles)

Extracts titles from `df` column Name. Then filters out titles that are not
in `least_occuring_titles` vector (those are set to "Rare." value). This
title vector is then added to DataFrame `df`. 
"""
function add_titles!(df::DataFrame, least_occuring_titles)
    matches = match.(r"\w+\.", df.Name)
    titles = [match.match for match in matches]
    new_titles = replace(x -> x in least_occuring_titles ? "Rare." : x, titles)
    df[!, :Titles] = new_titles
    return
end

"""
    compute_fare_mean(embarked, pclass, df)

Computes mean of fare in groups of given `embarked` value and `pclass` value
in DataFrame `df`.
"""
function compute_fare_mean(embarked, pclass, df)
    sel_fares_df = @from row in dropmissing(df, :Fare) begin
        @where row.Embarked == embarked && row.Pclass == pclass
        @select {
            row.Fare,
        }
        @collect DataFrame
    end

    fare_mean = mean(sel_fares_df.Fare)

    return fare_mean
end

