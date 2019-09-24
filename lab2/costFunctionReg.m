function [J, grad] = costFunctionReg(theta, X, y, lambda)
% COSTFUNCTION Вычисление функции стоимости и значения градиента(ов)для
% задачи логистической регрессии с регуляризацией
% J = COSTFUNCTION(theta, X, y, lambda) вычисляет функцию стоимости, используя
% theta в качестве параметра логистической регрессии, а также значение(я)
% градиентов

% Иницализация основных величин
m = length(y); % количество обучающих элементов

% В процессе выполнения задания, следующие переменные должны быть вычислены
% правильно 
J = 0;
grad = zeros(size(theta));

% ====================== ВАШ КОД ЗДЕСЬ ======================
h = sigmoid(X*theta);
J = sum(-y.*log(h)-(1-y).*log(1-h))./m + lambda*sum(theta.^2)/(2*m);

for j=1:length(theta)
    temp = 0;
    for i=1:m
        temp = temp + (h(i)-y(i))*X(i,j);
    end
    if j > 1
        grad(j) = temp + lambda*theta(j);
    else
        grad(j) = temp;
    end
end
grad = grad./m;
% =============================================================

end