% reset rand seed
rng('default');

% load training data
[num_sample, p] = size(train_X);

class_num = max(train_Y);

% parameters
iter_num = 1;
epoch = 1;
learning_rate = 1e3;
t0 = 1e4;
skip = 1e2;

% locally linear anchor points
anchors_num = 50;
nearest_neighbor = 5;

beta = 1;

loss_SAPL_test = zeros(iter_num, epoch);
loss_SAPL_train = zeros(iter_num, epoch);
accuracy_SAPL = zeros(iter_num, epoch);

for i=1:iter_num
    b = zeros(class_num, 1, anchors_num);
    W = zeros(class_num, p, anchors_num);
    loss_cumulative_llsvm = zeros(1, num_sample); 
    
    loss_cumulative_SAPL = zeros(1, num_sample);                                     
    
    % initial anchor points via K-means
    fprintf('Start K-means...\n');
    [~, anchors, ~, ~, ~] = litekmeans(train_X, anchors_num, 'Replicates', 10);
    fprintf('K-means done..\n');
    
%     fprintf('Start liblinear initialization...\n');
%     [W, b] = initial_paras(train_X, train_Y, anchors);
%     fprintf('liblinear initialization done..\n');
    
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
            y = -ones(1, class_num);
            y(Y_train(j,:)) = 1;
            
            % pick nearest anchor points
            [anchor_idx, weight] = knn(anchors, X, nearest_neighbor, beta);
            gamma = weight / sum(weight);
            
%             y_predict = zeros(1, class_num);
            y_anchor = zeros(class_num, nearest_neighbor);
            for m = 1:class_num
                y_anchor(m,:) = X * squeeze(W(m,:,anchor_idx)) + squeeze(b(m,:,anchor_idx))';
%                 y_predict(m) = gamma * y_anchor';
%                 if 1 - y(m) * y_predict(m) > 0
%                     y_anchor = y_anchor + y_anchor;
%                 end
                
            end
            
            y_predict = gamma * y_anchor';
            % hinge loss
            err = 1 - y .* y_predict;
            err(err<0) = 0;
            
            % cumulative training hinge loss
            idx = (t-1)*num_sample + j;
            if idx == 1
                loss_cumulative_SAPL(idx) = sum(err);
            else
                loss_cumulative_SAPL(idx) = (loss_cumulative_SAPL(idx-1) * (idx-1) + sum(err))/idx;
            end
            
            % record loss epoch-wise
            loss_SAPL_train(i, t) = loss_cumulative_SAPL(idx);
            err(err>0) = 1;
            
            % sgd update
%             if err > 0
%                 W(:,anchor_idx) = W(:,anchor_idx) + learning_rate / (idx + t0) * y * repmat(gamma,p,1) .* repmat(X',1,nearest_neighbor);
%                 b(anchor_idx) = b(anchor_idx) + learning_rate / (idx + t0) * y * gamma;
%                 
%                 % update anchor points (SAPL)
%                 s = 2 * beta * (repmat(X, nearest_neighbor, 1) - anchors(anchor_idx, :)).*repmat(weight, p, 1)';
%                 base = -s * sum(weight.*y_anchor);
%                 base = base + repmat(y_anchor',1,p).* s*sum(weight);
%                 anchors(anchor_idx,:) = anchors(anchor_idx,:) + learning_rate / (idx + t0) * (y* base/(sum(weight).^2));
%             end
            
            for m=1:class_num
                W(m,:,anchor_idx) = squeeze(W(m,:,anchor_idx)) + learning_rate / (idx + t0) * y(m)*err(m) * repmat(gamma,p,1) .* repmat(X',1,nearest_neighbor);
                b(m,:,anchor_idx) = squeeze(b(m,:,anchor_idx)) + learning_rate / (idx + t0) * y(m)*err(m) * gamma';
            end
            
            % update anchor points (SAPL)
            y_anchor = sum(y_anchor .* repmat((y.*err)',1,nearest_neighbor));
            s = 2 * beta * (repmat(X, nearest_neighbor, 1) - anchors(anchor_idx, :)).*repmat(weight, p, 1)';
            base = -s * sum(weight.*y_anchor);
            base = base + repmat(y_anchor',1,p).* s*sum(weight);
            anchors(anchor_idx,:) = anchors(anchor_idx,:) + learning_rate / (idx + t0) * (base/(sum(weight).^2));
            
            % regularization
            count = count - 1;
            if count <= 0
                W = W * (1 - skip/(idx + t0));
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
            y = -ones(1, class_num);
            y(test_Y(k,:)) = 1;
            
            [anchor_idx, weight] = knn(anchors, X, nearest_neighbor, beta);
            gamma = weight / sum(weight);
            
            y_predict = zeros(1, class_num);
            for m = 1:class_num
                y_anchor = X * squeeze(W(m,:,anchor_idx)) + squeeze(b(m,:,anchor_idx))';
                y_predict(m) = gamma * y_anchor';
            end
            
            err = 1 - y .* y_predict;
            
            [~,label] = max(y_predict);
            err(err<0) = 0;
            
            loss = loss + sum(err);
            
            % accuracy
            if label == test_Y(k,:)
                correct_num = correct_num + 1;
            end
        end
        
        % record test hinge loss epoch-wise
        loss_SAPL_test(i, t) = loss / num_sample_test;
        
        % record test accuracy epoch-wise
        accuracy_SAPL(i,t) = correct_num / num_sample_test;
        
        toc;
        fprintf('validation done\n');
       
    end
end


%% plot cumulative learning curve
plot(loss_cumulative_SAPL, 'DisplayName', 'LLC-SAPL');
legend('-DynamicLegend');
xlabel('Number of samples seen');
ylabel('Hinge loss');
grid on;

%% plot learning curve epoch-wise
hold on
plot(loss_SAPL_train(1,:),'r--o', 'DisplayName', 'LLC-SAPL');
legend('-DynamicLegend');
xlabel('epoch');
ylabel('Hinge loss');
title('Cumulative Learning Curve')
grid on;
