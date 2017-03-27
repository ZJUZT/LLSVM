function [W, b] = initial_paras(X, Y, centroids)
    
    % soft assignment -- get local coordinates
    gamma = EuDist2(X, centroids);
    
    % augument vector -- add bias
    [n, p] = size(X);
    [c,~] = size(centroids);
    
    new_p = (p+1)*c;
    new_X = zeros(n, new_p);
    for i=1:n
        x = X(1,:);
        tmp = [x 1];
        tmp = gamma(i,:)' * tmp;
        tmp = reshape(tmp,[1,new_p]);
        new_X(i,:) = tmp;
    end
    
    % liblinear
    model = liblinear_train(Y, sparse(new_X));
    W = reshape(model.w,[p+1, c]);
    b = W(end,:);
    W = W(1:end-1,:);
    
end