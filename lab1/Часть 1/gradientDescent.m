function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT �������������� ���������� (��������) ��������� theta
%   ������� ������������ ������
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) �������������� 
%   theta � �������� ���������� num_iters ����������� ��������
%   � ���������� �������� �������� alpha

% ������������� ����������
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== ��� ��� ����� ======================
    h = X*theta-y;
    theta = theta - (X'*h).*alpha./m;

    % ============================================================

    % ���������� ������� ��������� �� ������ ��������    
    J_history(iter) = computeCost(X, y, theta);

end

end
