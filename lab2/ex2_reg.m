%% ���� ���������������� ������� (�������� 2019 - ������ 2020)
%
%  ����������
%  ------------
% 
%  ���� ���� �������� ���, ������� �������� ��������� ������������ ������,
%  ���������� ������������� ���������. ������� ����������������� ���������
%  �������
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m
%
%  ��� ���������� ������� ��� ������������� �������� ����� ���� ����� ����,
%  ������� ������� � ������ ����� ��� � ���� ������, ������� �� ������� �
%  �������. ������, ��������� ��������� ���������� � �������� ������������� 
%  �������� (��������,lambda).
%

%% �������������
clear ; close all; clc

%% �������� ������
%  ������ ��� ������� �������� ��������� ������, � ������ ������� ��������
%  ����� ������

data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

plotData(X, y);

hold on;

xlabel('���� ��������� 1')
ylabel('���� ��������� 2')

legend('y = 1', 'y = 0')
hold off;


%% =========== ������� 1(5): ���������������� ������������� ��������� ============
%  ��������: ����������� ������ �� �������� ������� �������������� �, 
%  �������������, �� ����� ���� ��������� �� ������������� � ������������� 
%  ������ ������ ������. ������� ���������� ���������� ������ ������� 
%  ������������� ��������� �� �������� � ������ �������, ��������� �� 
%  ������������� ������������� ������� ������� ���� ��������.
%  ��� �� �����, ������������� ������������� ��������� ��������, � ������,
%  ����������� ���������� ��������������� ���������, ������� ��������������� ���������. 

% ���������� ��������������� ���������

% ��������: ������� mapFeature ��������� ������� ������

X = mapFeature(X(:,1), X(:,2));

% ������������� ��������
initial_theta = zeros(size(X, 2), 1);

% ������� ��������� ������������� ������ 1
lambda = 1;

% ���������� � ����������� ��������� �������� ������� ��������� �
% ��������� ��� ���������������� ������������� ���������

[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('�������� ������� ��������� ��� ��������� �������� theta (zeros): %f\n', cost);

fprintf('����� � ���������� ���������. ������� ����� ������� ��� �����������.\n');
pause;

%% ============= ������� 2(6): ������������� � �������� =============
%  �������������� �������:
%  � ���� �������, �������� �������� lambda, �������� ������� ������������� �� ��������� 
%  ������� ���������� �������
%
%  ��������: ������������������ ������ ��� lambda (0, 1, 10, 100).
%%

% ������������� ����������
initial_theta = zeros(size(X, 2), 1);

lambdas = [0,1,10,50,100,500];
for i =1:length(lambdas)
    % ������� lambda (��������������� ������������ ����� ���������)
    lambda = lambdas(i);

    % ������� �����
    options = optimset('GradObj', 'on', 'MaxIter', 400);

    % �����������

    %[theta, J, exit_flag] = ...
    %fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
    [theta, J, exit_flag] = ...
    	fmincg(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

    % ����������� ������� �������
    plotDecisionBoundary(theta, X, y);
    hold on;
    title(sprintf('lambda = %g', lambda))

    % �������
    xlabel('���� ��������� 1')
    ylabel('���� ��������� 2')

    legend('y = 1', 'y = 0', '�������')
    hold off;

    % ������������ �������� �������������� ������ ������
    p = predict(theta, X);

    fprintf('�������� ��������: %f\n', mean(double(p == y)) * 100);
end

