using Statistics

export predict, logistic_regression, get_best_λ, accuracy

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
    ##check shapes!
    probs = σ.(X * w)
    predictions = falses(size(X, 1))
    predictions[probs.>=0.5] .= true

    return predictions
end

"""
    gradient_descent(grad::Function, x; α=0.01, max_iter=1000)

Performs gradient descent using `grad` gradient function and startg point `x` using
learning rate `alpha` (default 0.01) for `max_iter` iterations (default 1000).
"""
function gradient_descent(grad::Function, x; α=0.01, max_iter=1000)
    for _ in 1:max_iter
        x -= α * grad(x)
    end

    return x
end

"""
    logistic_regression(X, y; λ=0, α=0.01, w=zeros(size(X, 2)), max_iter=1000)

Performs standard (if `λ` = 0) or ridge logistic regression (if `λ` != 0) using
data matrix `X`, true classes `y`, learning rate for gradient descent `alpha`,
initial weights `w` and maximum number of iterations for gradient descent `max_iter`.
"""
function logistic_regression(X, y; λ=0, α=0.01, w=zeros(size(X, 2)), max_iter=1000)
    #check shapes
    function grad(w)
        m = size(X, 1)
        y_pred = σ.(X * w)
        gr = X' * (y_pred - y) / m + λ * w / m
        return gr
    end

    w = gradient_descent(grad, w; max_iter=max_iter, α=α)
    return w
end

"""
    get_best_λ(X_train, y_train, X_val, y_val; max_λ=300)

Finds best parameter λ for ridge regression from interval <0,`max_λ`> by
trying logistic regression with found parameter on train part of dataset 
(`X_train`, `y_train`) and then compute error on development part of
dataset (`X_val`, `y_val`).
"""
function get_best_λ(X_train, y_train, X_val, y_val; max_λ=300)
    #check shapes
    λs = range(0, max_λ, step=0.1)
    best_λ = 0
    best_error = Inf
    for λ in λs
        w = logistic_regression(X_train, y_train; λ=λ)
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