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
%  �������.
%

%% �������������
clear ; close all; clc

%% �������� ������
%  ������ ��� ������� �������� ��������� ������, � ������� ������� ��������
%  ����� ������

data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);

%% ==================== ����� 1. ����������� ============
%  ������� ������������ ������, ����� ����������� ���� ������ � ����� 

fprintf(['����������� ������ � "+", ��������������� �������� (y = 1), � "o", ' ...
         '��������������� �������� (y = 0).\n']);

plotData(X, y);

fprintf('����� � ���������� ���������. ������� ����� ������� ��� �����������.\n');
pause;


%% ===== ������� 2: ���������� ������� ��������� � ���������� =============
%  � ���� ����� �������, ������������ ������� ��������� � ��������� ��� ������
%  ������������� ���������. ������� ��������� ��� ������� 
%  costFunction.m

%  ������� �������� �������
[m, n] = size(X);

X = [ones(m, 1) X];

initial_theta = zeros(n + 1, 1);

[cost, grad] = costFunction(initial_theta, X, y);

fprintf('�������� ������� ��������� ��� ��������� (�������) ��������� ������� thet�: %f\n', cost);
fprintf('�������� ��������� ��� ��������� (�������) ��������� ������� thet�: \n');
fprintf(' %f \n', grad);

fprintf('����� � ���������� ���������. ������� ����� ������� ��� �����������.\n');
pause;


%% == ������� 3: ����������� � �������������� ������� fminunc (fmincg)  ===
%  � ���� ������� �������������� ������ ��� ���������� ������ ������������ 
%  ������ ������������ ��� ���������� ����������� �������� ���������� theta 
%  ������������ ���������� ������� fminunc
%  ��������: � ��������� ������� ������ MATLAB������ ������� ���������� �
%  Optimization toolbox. ���� ������������� ���� ������� �����������
%  ��������������� � ������������ ������, ������������� ������������ ������
%  fmincg, ������� ��������� ����� ����������� ����������.
%  ��� ��������� fmincg ��������� � ������� ��������.

%  ������� ����� ��� fminunc (fmincg)
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  ���������� ��������� fminunc (fmincg) ��� ���������� �����������
%  �������� theta
%  ������� ���������� �������� ������� ��������� � theta 

%  ��������: ������� �������� ���������� �������

fprintf('��������!\n')
fprintf('..............................................................\n')
fprintf('� ���� ex2.m ������ ���� ������ ����� ����� fminunc ��� fmincg\n')
fprintf('������� ����� ������� ��� �����������\n');
fprintf('..............................................................\n')
pause;

[theta, cost] = ...
fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
%[theta, cost] = ...
%	fmincg(@(t)(costFunction(t, X, y)), initial_theta, options);

% ����� theta �� �����
fprintf('�������� theta, ��������� fmincg: %f\n', cost);
fprintf('�������� theta: \n');
fprintf(' %f \n', theta);

% ����������� ������� �������
plotDecisionBoundary(theta, X, y);


hold on;
% �������� ����
xlabel('���')
ylabel('��������������� ��������')

% "�������"
legend('�������', '��������')
hold off;

fprintf('����� � ���������� ���������. ������� ����� ������� ��� �����������.\n');
pause;

%% ============== ������� 4: ������������ � ������ �������� ==============
%  ����� ���������� �������� ������� ���������� ������������ ��� ���
%  ������, ������� �� ���� ������������ � �������� ��������, �.�. ������,
%  ������� ��������� � �������� ������������ ����������. 
%  � ���� �������, ������������� ��������� ������������ ��� ������������
%  ����������� ����, ��� ���������, ��������� �������� ���������������
%  ���������� �����������, ��� - 45 ��., �������� - 85 ��, ��������
%  ���������.
%
%  ����� ����, ��� ��������� ��������� �������� (�����������)
%  ������������ ������������� ������.
%
%  ������ ������� � ���������� ���� predict.m

%  ����������� ����������� ����������������� ��� ������ ���������, 
%  ��������� �������� ��������������� ���������� �����������, ��� - 45 ��., 
%  �������� - 85 ��.

prob = sigmoid([1 45 85] * theta);
fprintf(['��� ��������� � ������� ���� 45 � ��������� 85, ��������������� ������� ' ...
         '� ������������ %f\n\n'], prob);

% ������������ �������� �������������� ������ ������
p = predict(theta, X);

fprintf('�������� ��������: %f\n', mean(double(p == y)) * 100);

fprintf('����� � ���������� ���������. ������� ����� ������� ��� �����������.\n');
pause;

