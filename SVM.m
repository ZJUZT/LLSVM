% reset rand seed
rng('default');

% load training data
[num_sample, p] = size(train_X);

% parameters
iter_num = 1;
epoch = 10;
learning_rate = 1e-1;

loss_svm_test = zeros(iter_num, epoch);
loss_svm_train = zeros(iter_num, epoch);
accuracy_svm = zeros(iter_num, epoch);

for i=1:iter_num
    b = 0;
    W = zeros(1,p);
    loss_cumulative_svm = zeros(1, num_sample);                                     
    
    % shuffle
    re_idx = randperm(num_sample);
    X_train = train_X(re_idx,:);
    Y_train = train_Y(re_idx,:);
    
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
            
            y_predict = W*X' + b;
            
            % hinge loss
            err = 1 - y * y_predict;
            
            % cumulative training hinge loss
            idx = (t-1)*num_sample + j;
            if idx == 1
                loss_cumulative_svm(idx) = max(0,err);
            else
                loss_cumulative_svm(idx) = (loss_cumulative_svm(idx-1) * (idx-1) + max(0,err))/idx;
            end
            
            % record loss epoch-wise
            loss_svm_train(i, t) = loss_cumulative_svm(idx);
            
            % sgd update
            if err > 0
                W = W + learning_rate*y * X;
                b = b + learning_rate*y;
            end
        end
        
        % validate epoch-wise
        loss = 0.0;
        correct_num = 0;
        fprintf('validating\n');
        tic;
        [num_sample_test, ~] = size(test_X);
        
        for k=1:num_sample_test
            
            if mod(k,1e5)==0
                fprintf('%d epoch(validation)---processing %dth sample\n',i, k);
            end
            
            X = test_X(k,:);
            y = test_Y(k,:);
            
            y_predict = W*X' + b;
            err = 1 - y * y_predict;
            loss = loss + max(0, err);
            
            % accuracy
            if (y_predict>=0 && y==1) || (y_predict<0&&y==-1)
                correct_num = correct_num + 1;
            end
        end
        
        % record test hinge loss epoch-wise
        loss_svm_test(i, t) = loss / num_sample_test;
        
        % record test accuracy epoch-wise
        accuracy_svm(i,t) = correct_num / num_sample_test;
        
        toc;
        fprintf('validation done\n');
       
    end
end


%% plot cumulative learning curve
plot(loss_cumulative_svm, 'DisplayName', 'Linear SVM');
legend('-DynamicLegend');
xlabel('Number of samples seen');
ylabel('Hinge loss');
grid on;


