using CSV
using DataFrames
using Statistics

export categorical_to_one_hot, standardize

function categorical_to_one_hot(vec::AbstractVector{<:AbstractString})
    unique_vals = unique(vec)
    ret_mat = zeros(Bool, (length(vec), length(unique_vals)))
    
    for i in eachindex(unique_vals)
        ret_mat[vec.==unique_vals[i], i] .= true
    end

    return ret_mat
end

function standardize(X::Matrix{<:Number})
    col_mean = mean(X; dims=1)
    col_std = std(X; dims=1)
    return (X .- col_mean) ./ col_std
end