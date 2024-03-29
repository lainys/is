function word_indices = processEmail(email_contents)
% PROCESSEMAIL ������������ ������������� ������������ ������
% � ���������� ������ �������� ���� (word_indices) 
%

% �������� �������
vocabList = getVocabList();

% ������������� ������������ ����������
word_indices = [];

% ================== ������������� ������������ ������ ====================

% ���������� � �������� ����������
% ������� ����������������� ��������� ������ ����, ���� � �������� ���������
% ��������� "����� ������" � ������� �����������

% hdrstart = strfind(email_contents, ([char(10) char(10)]));
% email_contents = email_contents(hdrstart(1):end);

% ��������� ��������
email_contents = lower(email_contents);

% �������� ���� HTML �����
% ��������������� ���������, ������� ���������� � ������� < � ������������� �������� >,
% ����� ���� �������� ���� ����� ������

email_contents = regexprep(email_contents, '<[^<>]+>', ' ');

% ��������� �������� ������
% ��������������� ���� ���������, ���������� ����� �� 0 �� 9
email_contents = regexprep(email_contents, '[0-9]+', 'number');

% ��������� URL-�������
% �������������� ���������, ����������  http:// ��� https://
email_contents = regexprep(email_contents, ...
                           '(http|https)://[^\s]*', 'httpaddr');

% ������������ ��������� Email �������
% ��������������� �������� ������, ���������� @ � ��������
email_contents = regexprep(email_contents, '[^\s]+@[^\s]+', 'emailaddr');

% �������������� ����� $
email_contents = regexprep(email_contents, '[$]+', 'dollar');


% ========================== ����������� ������ ������ ===========================
% ����������� ������ (tokenization - ��������������� ������, 
% lexical analysis - ����������� ������) - ��������� � ������ ����, ����� � ������ �������. 
% ��������, ���������� ������ �����������. 


fprintf('\n==== �������������� ������ ====\n\n');

l = 0;

while ~isempty(email_contents)

    [str, email_contents] = ...
       strtok(email_contents, ...
              [' @$/#.-:&*+=[]?!(){},''">_<;%' char(10) char(13)]);
   
    % �������� ���� �� ����������������� �������������������
    str = regexprep(str, '[^a-zA-Z0-9]', '');

    % ���������� ��������� "���������". �������� ��� ������� ���������� ������ ����� 
    % ��� ��������� ��������� �����. 
    % ����� ������������ ������� ������� � �������� ���������, �������������� �������� �������� � 1980 ����.
    try str = porterStemmer(strtrim(str)); 
    catch str = ''; continue;
    end;

    % �� ������������ ������� �������� �����
    if length(str) < 1
       continue;
    end

    % ====================== ��� ��� ����� ======================
    % ��������:     ��������� ����������������� �������, ����������� ������
    %               str � ���������� word_indices, ���� ����� �������������� � �������.
    %               ������� ������� ���� ���������� ����� � ������� ������. 
    %               ������� ������ (����. lookup table) � ��� ��������� ������, 
    %               ������ ������ ��� ������������� ������, ������������ � ����� 
    %               �������� ���������� �� �������� �������� ������.         
    %               ������� ���������� ����������� str � ������� vocabList. 
    %               ���� ���������� ������������, ������� �������� ������ 
    %               ����� � ������ ��������.
    %               ��������, ���� str = 'action', ����� ������� ���������� ��� 
    %               ����� � vocabList � ����, ��������, � vocabList{18} =
    %               'action', ����� ������� �������� 18 � ������� word_indices 
    %               (word_indices = [word_indices ; 18]; ).
    % 
    % ��������:  ������� vocabList{idx} ���������� � ����� � �������� idx � �������.
    % 
    % ���������: ����� ������������ �������  strcmp(str1, str2) ���
    % ��������� ���� �����(str1 � str2). �������� 1 ����� �������� ���� � 
    % ������ ��������������� ���� ������������������� ��������.
    %
    
    
    for i=1:length(vocabList)
        if strcmp(vocabList{i},str) == 1
            word_indices = [word_indices ; i];
            break
        end
    end
    
    
end
fprintf('\n\n=========================\n');

end
