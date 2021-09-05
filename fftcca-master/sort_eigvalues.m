function [eigvector, eigvalue] = sort_eigvalues(eigvector, eigvalue,sorttype)
%   According to the eigenvalues, sort the eigenvalue matrix and the eigenvector matrix 
    narginchk(2, 3);
    if nargin < 3
        sorttype = 'ascend';
    end
    nargoutchk(1, 2);
    eigvalue = diag(eigvalue);
    % Diagonal eigvalues to vector
    % Complex eigvalues to real
    [~, index] = sort(eigvalue,sorttype);
    eigvalue = eigvalue(index); % Sort as index                                                                      
    eigvector = eigvector(:,index);

end

