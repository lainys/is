function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE ������������ ��������� � ������ X 
%   FEATURENORMALIZE(X) ���������� ������������� ������ ������ X, � �������
%   ������� �������� ������� �������� 0 � ����������� ���������� -
%   1. ����� ��� �������� ������� ��������������� ����������
%   ��� ������ � ������� � �������� ��������� �������� (�����-�������� 
%   ������� ������).

% ���������� ��������� ���������� (�������) �������� ���� ����������
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== ��� ��� ����� ======================
% ����������:   �������, ��� ������� �������� �� ������ ������ ��������� 
%               ��� ������� �������� � ����������� ��� ��������� �� ������ ������,
%               ��� ���� ������� �������� ��������� ��� mu. �����, ����������� 
%               ����������� (��������������������) ���������� sigma
%               ��� ������� �������� � ��������� X-mu �� sigma.
%                
%
%               �������, ��� X ������������ ����� �������, � ������� ������
%               ������� ������������� �������, � ������ ������ - ��������� ������.
%               ������������ ������ ������� ����������� �������� ��� �������
%               �������� 
%
% ��������:     �� ������ �������� ������� ��� ���������� mu � sigma ��������������,
%               � ������ ��������������� ����������� ��������� MATLAB
%               'mean' � 'std':
%               http://matlab.exponenta.ru/ml/book2/chapter8/mean.php
%               http://matlab.exponenta.ru/ml/book2/chapter8/std.php
%     
mu = mean(X,1);
sigma = std(X,1);
X_norm = (X_norm - mu)./sigma;
% ============================================================

end
