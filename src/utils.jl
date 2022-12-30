using CSV
using DataFrames

export read_csv_to_df, save_df_to_csv, compute_class_error

function read_csv_to_df(file_path::String)
    if ispath(file_path)
        return CSV.read(file_path, DataFrame; header=true)
    else
        error("file not found")
    end
    
end

function save_df_to_csv(df::DataFrame, file_path::String)
    CSV.write(file_path, df)
end

function compute_class_error(true_vals::BitVector, predicted_vals::BitVector)
    n_wrong = sum(xor.(true_vals, predicted_vals))
    return n_wrong / length(true_vals)
end

compute_class_error(true_vals::Vector{<:Int}, predicted_vals::Vector{<:Int}) = compute_class_error(Bool.(true_vals), Bool.(predicted_vals))
compute_class_error(true_vals::BitVector, predicted_vals::Vector{<:Int}) = compute_class_error(true_vals, Bool.(predicted_vals))
compute_class_error(true_vals::Vector{<:Int}, predicted_vals::BitVector) = compute_class_error(Bool.(true_vals), predicted_vals)