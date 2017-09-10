function [ model, metric ] = llsvm( training, validation, pars )
%FM Summary of this function goes here
%   Detailed explanation goes here

    train_X = training.train_X;
    train_Y = training.train_Y;
    
    test_X = validation.test_X;
    test_Y = validation.test_Y;

    [num_sample, p] = size(train_X);

    % parameters
    iter_num = pars.iter_num;
    learning_rate = pars.learning_rate;
    skip = pars.skip;
    count = skip;
    t0 = pars.t0;

    epoch = pars.epoch;

    beta = pars.beta;

    % locally linear anchor points
    anchors_num = pars.anchors_num; 
    nearest_neighbor = pars.nearest_neighbor;

    loss_fm_test = zeros(iter_num, epoch);
    loss_fm_train = zeros(iter_num, epoch);
    accuracy_fm = zeros(iter_num, epoch);

    for i=1:iter_num

        tic;

        w0 = pars.w0;
        W = pars.W;
%         V = pars.V;
        
        re_idx = randperm(num_sample);
        X_train = train_X(re_idx,:);
        Y_train = train_Y(re_idx);

        % initial anchor points via K-means
        fprintf('Start K-means...\n');
        [~, anchors, ~, ~, ~] = litekmeans(train_X, anchors_num, 'Replicates', 1);
        fprintf('K-means done..\n');

        for t=1:epoch

            loss = 0;
            for j=1:num_sample

                X = X_train(j,:);
                y = Y_train(j,:);

                idx = (t-1)*num_sample + j;
                
                % pick nearest anchor points
                [anchor_idx, weight] = knn(anchors, X, nearest_neighbor, beta);
                gamma = weight / sum(weight);
                
                y_anchor = X * W(:,anchor_idx) + w0(anchor_idx);
                y_predict = gamma * y_anchor';
                % SGD update
                err = max(0, 1-y*y_predict);
                loss = loss + err;
                
                if err > 0
                    W(:,anchor_idx) = W(:,anchor_idx) + learning_rate / (idx + t0) * y * repmat(gamma,p,1) .* repmat(X',1,nearest_neighbor);
                    w0(anchor_idx) = w0(anchor_idx) + learning_rate / (idx + t0) * y * gamma;
                end

                % regularization
                count = count - 1;
                if count <= 0
                    W = W * (1 - skip/(idx + t0));
                    count = skip;
                end

            end

            loss_fm_train(i,t) = loss / num_sample;
            fprintf('[iter %d epoch %2d]---train loss:%.4f\t',i, t, loss_fm_train(i,t));

            % validate
            loss = 0;
            correct_num = 0;
            [num_sample_test, ~] = size(test_X);
            for k=1:num_sample_test

                X = test_X(k,:);
                y = test_Y(k,:);

                [anchor_idx, weight] = knn(anchors, X, nearest_neighbor, beta);
                gamma = weight / sum(weight);
                
                y_anchor = X * W(:,anchor_idx) + w0(anchor_idx);
                y_predict = gamma * y_anchor';
                err = max(0, 1-y_predict*y);
                loss = loss + err;

                if (y_predict>=0 && y==1) || (y_predict<0&&y==-1)
                    correct_num = correct_num + 1;
                end

            end

            loss_fm_test(i,t) = loss / num_sample_test;
            fprintf('test loss:%.4f\t', loss_fm_test(i,t));
            accuracy_fm(i,t) = correct_num/num_sample_test;
            fprintf('\ttest accuracy:%.4f', accuracy_fm(i,t));

            fprintf('\n');

        end
        
        toc;
    end
    
    % pack output
    % model
    model.w0 = w0;
    model.W = W;
    
    % metric
    metric.loss_train = loss_fm_train;
    metric.loss_test = loss_fm_test;
    metric.loss_accuracy = accuracy_fm;

end