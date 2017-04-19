% reset rand seed
rng('default');

% load training data
[num_sample, p] = size(train_X);

class_num = max(train_Y);

% parameters
iter_num = 1;  
epoch = 10;
learning_rate = 1e2;
t0 = 1e4;
skip = 1e2;
    
% locally linear anchor points
anchors_num = 50;

LC = 5;

loss_JO_test = zeros(iter_num, epoch);
loss_JO_train = zeros(iter_num, epoch);
accuracy_JO = zeros(iter_num, epoch);

nn_train = zeros(iter_num, epoch);

for i=1:iter_num
    
    num_nn = 0;
    minmum_K = 100;
    maximum_K = 0;
    
    b = zeros(class_num, 1, anchors_num);
    W = zeros(class_num, p, anchors_num);
    loss_cumulative_llsvm = zeros(1, num_sample); 
    loss_cumulative_JO = zeros(1, num_sample);                                     
    
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
        num_nn_batch = 0;
        nn_epoch = 0;
        tic;
        for j=1:num_sample
            if mod(j,1e3)==0
                toc;
                fprintf('%d iter(%d epoch)---processing %dth sample\n', i, t, j);
                fprintf('batch average value of K in KNN is %.2f\n', num_nn_batch/1e3);
                fprintf('overall average value of K in KNN is %.2f\n', num_nn/((t-1)*num_sample+j));
                
                num_nn_batch = 0;
                tic;
            end
            
            X = X_train(j,:);
            y = -ones(1, class_num);
            y(Y_train(j,:)) = 1;
            
            % pick nearest anchor points (adaptive KNN)
            [anchor_idx, D, lam] = Dynamic_KNN(anchors, X, LC);
            nearest_neighbor = length(anchor_idx);
            weight = lam - D;
            gamma = weight / sum(weight);
            
            y_predict = zeros(1, class_num);
            for m = 1:class_num
                y_anchor = X * squeeze(W(m,:,anchor_idx)) + squeeze(b(m,:,anchor_idx))';
                y_predict(m) = gamma * y_anchor';
            end
            
            if minmum_K>nearest_neighbor
                minmum_K=nearest_neighbor;
            end

            if maximum_K<nearest_neighbor
                maximum_K=nearest_neighbor;
            end
            
            num_nn = num_nn + nearest_neighbor;
            num_nn_batch = num_nn_batch + nearest_neighbor;
            
            nn_epoch = nn_epoch + nearest_neighbor;
            
            % hinge loss
            err = 1 - y .* y_predict;
            err(err<0) = 0;
            
            % cumulative training hinge loss
            idx = (t-1)*num_sample + j;
            if idx == 1
                loss_cumulative_JO(idx) = max(0,err);
            else
                loss_cumulative_JO(idx) = (loss_cumulative_JO(idx-1) * (idx-1) + max(0,err))/idx;
            end
            
            % record loss epoch-wise
            loss_JO_train(i, t) = loss_cumulative_JO(idx);
            err(err>0) = 1;
            
            % sgd update
%             if err > 0
%                 W(:,anchor_idx) = W(:,anchor_idx) + learning_rate / (idx + t0) * y * repmat(gamma,p,1) .* repmat(X',1,nearest_neighbor);
%                 b(anchor_idx) = b(anchor_idx) + learning_rate / (idx + t0) * y * gamma;
%                 
%                 % update anchor points (SAPL)
%                 s = 2 * LC * (repmat(X, nearest_neighbor, 1) - anchors(anchor_idx, :));
%                 base = -s * sum(weight.*y_anchor);
%                 base = base + repmat(y_anchor',1,p).* s*sum(weight);
%                 anchors(anchor_idx,:) = anchors(anchor_idx,:) + learning_rate / (idx + t0) * (y* base/(sum(weight).^2));
%             end
            
            for m=1:class_num
                W(m,:,anchor_idx) = squeeze(W(m,:,anchor_idx)) + learning_rate / (idx + t0) * y(m)*err(m) * repmat(gamma,p,1) .* repmat(X',1,nearest_neighbor);
                b(m,:,anchor_idx) = squeeze(b(m,:,anchor_idx)) + learning_rate / (idx + t0) * y(m)*err(m) * gamma';
            end
            
            % update anchor points (SAPL)
                s = 2 * LC * (repmat(X, nearest_neighbor, 1) - anchors(anchor_idx, :));
                base = -s * sum(weight.*y_anchor);
                base = base + repmat(y_anchor',1,p).* s*sum(weight);
                anchors(anchor_idx,:) = anchors(anchor_idx,:) + learning_rate / (idx + t0) * (sum(y.*err)* base/(sum(weight).^2));
            
            % regularization
            count = count - 1;
            if count <= 0
                W = W * (1 - skip/(idx + t0));
                count = skip;
            end
            
        end
        
        nn_train(i, t) = nn_epoch/num_sample;
        
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
            
            [anchor_idx, D, lam] = Dynamic_KNN(anchors, X, LC);
            nearest_neighbor = length(anchor_idx);
            weight = lam - D;
            
            gamma = weight / sum(weight);
            
%             y_anchor = X * W(:,anchor_idx) + b(anchor_idx);
%             y_predict = gamma * y_anchor';
            
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
        loss_JO_test(i, t) = loss / num_sample_test;
        
        % record test accuracy epoch-wise
        accuracy_JO(i,t) = correct_num / num_sample_test;
        
        toc;
        fprintf('validation done\n');
       
    end
end


%% plot cumulative learning curve
plot(loss_cumulative_JO, 'DisplayName', 'LLC-SAPL');
legend('-DynamicLegend');
xlabel('Number of samples seen');
ylabel('Hinge loss');
grid on;


%% plot learning curve epoch-wise
plot(loss_JO_train(1,:),'b--o', 'DisplayName', 'LLC-JO');
legend('-DynamicLegend');
xlabel('epoch');
ylabel('Hinge loss');
title('Cumulative Learning Curve')
grid on;

%% plot nn epoch-wise
plot(nn_train,'b--o', 'DisplayName', 'nn');
legend('-DynamicLegend');
xlabel('epoch');
ylabel('nn');
title('Average NN number Learning Curve')
grid on;