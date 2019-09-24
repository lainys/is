function [C, sigma] = dataset3Params(X, y, Xval, yval)
%   DATASET3PARAMS ���������� ������� ��������� C � sigma ��� ������� �����
%   ����������, � ������� ��������� ���������� ����������� �������� (C, sigma) 
%   ��� ������������ ������������� SVM � ��������� ���������� �������� ��������
%   (��������, c ����������� �����)
%   
%   [C, sigma] = EX5PARAMS(X, y, Xval, yval) ���������� C � sigma, 
%   ��������������� ������ ������.
%   ������� ����������������� �������, ��������� ����� ������������ �������� (�����-���������).
%

% ������� ��������� ��������� ��������� ��������.

C = 1;
sigma = 0.3;

% ====================== ��� ��� ����� ======================
% ��������: ����� ������������ ����������� �������, ��������� � ��������: 
%           predictions = svmPredict(model, Xval).
%
% ��������: ���������� ����� ���������� ������ ��� ������ ������, ��������� ��� ��������. 
%           ������ ���������� ���� �������� ��� ������������ ��������, ������������������ 
%           �����������. � Matlab ������ ������ ����� ���������, ��������� ������� ���������� 
%           ��������  mean(double(predictions ~= yval))
figure;
values = [0.01 0.03 0.1 0.3 1 3 10 30 100 300];
%values = [0.01 0.03 1 3];
n = length(values);
acc_best = zeros(n);
for i = 1:n
    for j = 1:n
        C_i = values(i);
        sigma_i = values(j);
        subplot(n,n,(i-1)*n+j)
        model= svmTrain(X, y, C_i, @(x1, x2) gaussianKernel(x1, x2, sigma_i)); 
        visualizeBoundary(Xval, yval, model);
        predictions = svmPredict(model, Xval);
        acc_best(i,j) = mean(double(predictions ~= yval));
        title(sprintf('C = %g,sigma = %g', C_i,sigma_i))
    end
end
[~,i_ind] = min(min(acc_best,[],2));
[~,j_ind] = min(min(acc_best,[],1));
C = values(i_ind);
sigma = values(j_ind);
% =========================================================================

end
