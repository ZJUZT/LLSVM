% reset rand seed
rng('default');

% load training data
[num_sample, p] = size(train_X);

% parameters
iter_num = 1;
epoch = 10;
learning_rate = 1e4;

t0 = 1e4;
skip = 1e3 ;

% locally linear anchor points
anchors_num = 100;
nearest_neighbor = 8;

beta = 1.0;

loss_llsvm_test = zeros(iter_num, epoch);
loss_llsvm_train = zeros(iter_num, epoch);
accuracy_llsvm = zeros(iter_num, epoch);

for i=1:iter_num
    b = zeros(1, anchors_num);
    W = zeros(p, anchors_num);
    loss_cumulative_llsvm = zeros(1, num_sample);                                     
    
    % initial anchor points via K-means
    fprintf('Start K-means...\n');
    [~, anchors, ~, ~, ~] = litekmeans(train_X, anchors_num,'MaxIter', 100, 'Replicates', 1);
    fprintf('K-means done..\n');
    
    % shuffle
    re_idx = randperm(num_sample);
    X_train = train_X(re_idx,:);
    Y_train = train_Y(re_idx,:);
    
    count = skip;
    
    for t=1:epoch
        tic;
        
        for j=1:num_sample
            if mod(j,1e3)==0
                toc;
                fprintf('%d iter(%d epoch)---processing %dth sample\n', i, t, j);
                tic;
            end
            
            X = X_train(j,:);
            y = Y_train(j,:);
            
            % pick nearest anchor points
            [anchor_idx, weight] = knn(anchors, X, nearest_neighbor, beta);
            gamma = weight / sum(weight);
            
            y_anchor = X * W(:,anchor_idx) + b(anchor_idx);
            y_predict = gamma * y_anchor';
            
            % hinge loss
            err = 1 - y * y_predict;
            
            % cumulative training hinge loss
            idx = (t-1)*num_sample + j;
            if idx == 1
                loss_cumulative_llsvm(idx) = max(0,err);
            else
                loss_cumulative_llsvm(idx) = (loss_cumulative_llsvm(idx-1) * (idx-1) + max(0,err))/idx;
            end
            
            % record loss epoch-wise
            loss_llsvm_train(i, t) = loss_cumulative_llsvm(idx);
            
            % sgd update
            if err > 0
                W(:,anchor_idx) = W(:,anchor_idx) + learning_rate / (idx + t0) * y * repmat(gamma,p,1) .* repmat(X',1,nearest_neighbor);
                b(anchor_idx) = b(anchor_idx) + learning_rate / (idx + t0) * y * gamma;
            end
            
            % regularization
            count = count - 1;
            if count <= 0
                W(:,anchor_idx) = W(:,anchor_idx) * (1 - skip/(idx + t0));
                count = skip;
            end
        end
        
        % validate epoch-wise
        loss = 0.0;
        correct_num = 0;
        fprintf('validating\n');
        tic;
        [num_sample_test, ~] = size(test_X);
        
        for k=1:num_sample_test
            
            if mod(k,1e4)==0
                fprintf('%d epoch(validation)---processing %dth sample\n',i, k);
            end
            
            X = test_X(k,:);
            y = test_Y(k,:);
            
            [anchor_idx, weight] = knn(anchors, X, nearest_neighbor, beta);
            gamma = weight / sum(weight);
            
            y_anchor = X * W(:,anchor_idx) + b(anchor_idx);
            y_predict = gamma * y_anchor';
            
            err = 1 - y * y_predict;
            
            loss = loss + max(0, err);
            
            % accuracy
            if (y_predict>=0 && y==1) || (y_predict<0&&y==-1)
                correct_num = correct_num + 1;
            end
        end
        
        % record test hinge loss epoch-wise
        loss_llsvm_test(i, t) = loss / num_sample_test;
        
        % record test accuracy epoch-wise
        accuracy_llsvm(i,t) = correct_num / num_sample_test;
        
        toc;
        fprintf('validation done\n');
       
    end
end


%% plot cumulative learning curve
plot(loss_cumulative_llsvm, 'DisplayName', 'LLSVM');
legend('-DynamicLegend');
xlabel('Number of samples seen');
ylabel('Hinge loss');
grid on;


