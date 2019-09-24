function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI ����������� �������� theta ������� ������������ ������
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) ���������� ��������������� 
%   theta � �������� ���������� num_iters ������������ ����� � ���������� �������� �������� 
%   alpha

% ������������� ����������
m = length(y); % ���������� �������� ���������� ������
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== ��� ��� ����� ======================

    h = X*theta-y;
    theta = theta - (X'*h).*alpha./m;
    
    % ============================================================

   % ���������� ������� ��������� �� ������ ��������
    J_history(iter) = computeCostMulti(X, y, theta);
end

end
