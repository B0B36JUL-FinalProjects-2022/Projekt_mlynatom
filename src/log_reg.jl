using Statistics

export predict, logistic_regression, get_best_λ, accuracy, Adam_s, GD


"""
    σ(z)

Standard sigmoid function.

# Example

```julia-repl
julia> z = 1.0
1.0

julia> σ(z)
0.7310585786300049

```
"""
σ(z) = 1 / (1 + exp(-z))

"""
    predict(X::AbstractMatrix{<:Number}, w::AbstractVector{<:Number})

Predicts values for input matrix `X` given weights `w`.
"""
function predict(X::AbstractMatrix{<:Number}, w::AbstractVector{<:Number})
    if size(w)[1] != size(X)[2]
        throw(DimensionMismatch)
    end

    probs = σ.(X * w)
    predictions = falses(size(X, 1))
    predictions[probs.>=0.5] .= true

    return predictions
end


#abstract type for step in gradient_descent
abstract type Step end

#struct for standard gradient descent
struct GD <: Step
    α::Float64
end

#struct for Adam optimizer
mutable struct Adam_s <: Step
    α::Float64
    β1::Float64
    β2::Float64
    ϵ::Float64
    m
    v
end

"""
    function compute_step(step::GD, grad::Function, x, iter)

Compute step of gradient descent
"""
function compute_step(step::GD, grad::Function, x, iter)
    return -step.α * grad(x)
end

"""
    function compute_step(step::Adam, grad::Function, x, iter)

Compute step of Adam optimizer.
"""
function compute_step(step::Adam_s, grad::Function, x, iter)
    dx = grad(x)
    step.m = step.β1 .* step.m + (1 - step.β1) .* dx
    step.v = step.β2 .* step.v + (1 - step.β2) .* dx.^2

    m = step.m ./ (1 - step.β1^iter)
    v = step.v ./ (1 - step.β2^iter)

    return -step.α * m ./ (.√v .+ step.ϵ)
end

"""
    function gradient_descent(grad::Function, x, step::Step; max_iter=1000)

Performs gradient descent using `grad` gradient function and startg point `x` using
step `step` (struct with parameters) for `max_iter` iterations (default 1000).
"""
function gradient_descent(grad::Function, x, step::Step; max_iter=1000)
    for i in 1:max_iter
        x += compute_step(step, grad, x, i)
    end

    return x
end

"""
    function logistic_regression(X, y, step::Step; λ=0, w=zeros(size(X, 2)), max_iter=1000)

Performs standard (if `λ` = 0) or ridge logistic regression (if `λ` != 0) using
data matrix `X`, true classes `y`, step structure of parameters `step`,
initial weights `w` and maximum number of iterations for gradient descent `max_iter`.
"""
function logistic_regression(X, y, step::Step; λ=0, w=zeros(size(X, 2)), max_iter=10000)
    if size(w)[1] != size(X)[2]
        throw(DimensionMismatch)
    end

    function grad(w)
        m = size(X, 1)
        y_pred = σ.(X * w)
        gr = X' * (y_pred - y) / m + λ * w / m
        return gr
    end

    w = gradient_descent(grad, w, step; max_iter=max_iter)
    return w
end

"""
    function get_best_λ(X_train, y_train, X_val, y_val, step; max_λ=300)

Finds best parameter λ for ridge regression from interval <0,`max_λ`> by
trying logistic regression with found parameter on train part of dataset 
(`X_train`, `y_train`) and then compute error on development part of
dataset (`X_val`, `y_val`).
"""
function get_best_λ(X_train, y_train, X_val, y_val, step; max_λ=300)
    λs = range(0, max_λ, step=0.1)
    best_λ = 0
    best_error = Inf
    for λ in λs
        w = logistic_regression(X_train, y_train, step; λ=λ)
        preds = predict(X_val, w)
        error = compute_class_error(y_val, preds)
        if error < best_error
            best_error = error
            best_λ = λ
        end
    end

    return best_λ
end

"""
    accuracy(w, X, y)

Computes accuracy using given weights `w` on data matrix `X` and true classes `y`.
"""
function accuracy(w, X, y)
    return mean(predict(X, w) .== y)
end