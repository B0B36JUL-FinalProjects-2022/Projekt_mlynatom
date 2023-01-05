using CSV
using DataFrames

export read_csv_to_df, save_my_submission, compute_class_error

"""
    read_csv_to_df(file_path::String)

Reads csv file `file_path` and returns DataFrame.
"""
function read_csv_to_df(file_path::String)
    if ispath(file_path)
        return CSV.read(file_path, DataFrame; header=true)
    else
        throw(ErrorException("file not found"))
    end
    return
end

"""
    save_df_to_csv(df::DataFrame, file_path::String)

Saves dataframe `df` to file on `file_path`.
"""
function save_df_to_csv(df::DataFrame, file_path::String)
    CSV.write(file_path, df)
    return
end

"""
    save_my_submission(predictions, ids::Vector{<:Int}; file_path::String="data/my_submission.csv")

Saves predictions `predictions` with indices `ids` to file `file_path`.
"""
function save_my_submission(predictions, ids::Vector{<:Int}; file_path::String="data/my_submission.csv")
    new_df = DataFrame(PassengerId=ids, Survived=Int8.(predictions))
    save_df_to_csv(new_df, file_path)
    return
end

"""
    compute_class_error(true_vals::BitVector, predicted_vals::BitVector)

Computes error of prediction `predicted_vals` against truth `true_vals`.

```julia-repl
julia> compute_class_error([1,0,0], [1,1,0])
0.3333333333333333
```
"""
function compute_class_error(true_vals::BitVector, predicted_vals::BitVector)
    n_wrong = sum(xor.(true_vals, predicted_vals))
    return n_wrong / length(true_vals)
end

compute_class_error(true_vals::Vector{<:Int}, predicted_vals::Vector{<:Int}) = compute_class_error(Bool.(true_vals), Bool.(predicted_vals))
compute_class_error(true_vals::BitVector, predicted_vals::Vector{<:Int}) = compute_class_error(true_vals, Bool.(predicted_vals))
compute_class_error(true_vals::Vector{<:Int}, predicted_vals::BitVector) = compute_class_error(Bool.(true_vals), predicted_vals)