function [theta] = normalEqn(X, y)
%NORMALEQN ���������� ������� ���������� ��������� ��� ������� ������ �������� 
%   ���������� � ����������� ���������� 
%   NORMALEQN(X,y) � ������� ������ ������� �������������� ������� ������ �������� 
%   ���������� � ����������� ���������� � ������� ������� ���������� ��������� 

theta = zeros(size(X, 2), 1);

% ============= ��� ��� ����� ==============
% ����������: ����������� ������� pinv ��� ���������� �������� thea
%             
%
theta = inv(X'*X)*X'*y;

% ============================================================

end
