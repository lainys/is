function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Осуществляется вычисление (обучение) параметра theta
%   методом градиентного спуска
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) переопределяет 
%   theta в процессе выполнения num_iters градиентных итераций
%   с параметром скорости обучения alpha

% Инициализация переменных
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== ВАШ КОД ЗДЕСЬ ======================
    h = X*theta-y;
    theta = theta - (X'*h).*alpha./m;

    % ============================================================

    % Сохраняйте функцию стоимости на каждой итерации    
    J_history(iter) = computeCost(X, y, theta);

end

end
