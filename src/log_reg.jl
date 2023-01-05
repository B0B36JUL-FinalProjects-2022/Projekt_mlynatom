using Statistics

export predict, logistic_regression, get_best_λ

σ(z) = 1 / (1 + exp(-z))

function predict(X, w)
    probs = σ.(X * w)
    predictions = falses(size(X, 1))
    predictions[probs.>=0.5] .= true

    return predictions
end

function gradient_descent(grad::Function, x; α=0.01, max_iter=1000)
    for _ in 1:max_iter
        x -= α * grad(x)
    end

    return x
end

function logistic_regression(X, y; λ=0, α=0.01, w=zeros(size(X, 2)), max_iter=1000)

    function grad(w)
        m = size(X, 1)
        y_pred = σ.(X * w)
        gr = X' * (y_pred - y) / m + λ * w / m
        return gr
    end

    w = gradient_descent(grad, w; max_iter=max_iter, α=α)
    return w
end

function get_best_λ(X_train, y_train, X_val, y_val)
    λs = range(0, 300, step=0.1)
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