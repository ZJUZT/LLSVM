function [ model, metric ] = llc_jo( training, validation, pars )
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
    learning_rate_anchor = pars.learning_rate_anchor;
    skip = pars.skip;
    count = skip;
    t0 = pars.t0;

    epoch = pars.epoch;

    LC = pars.LC;

    % locally linear anchor points
    anchors_num = pars.anchors_num; 
    % nearest_neighbor = pars.nearest_neighbor;

    loss_fm_test = zeros(iter_num, epoch);
    loss_fm_train = zeros(iter_num, epoch);
    accuracy_fm = zeros(iter_num, epoch);

    max_nn_test = zeros(iter_num, epoch);
    min_nn_test = zeros(iter_num, epoch);
    nn_avg_test = zeros(iter_num, epoch);

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
                % pick nearest anchor points (adaptive KNN)
                [anchor_idx, D, lam] = Dynamic_KNN(anchors, X, LC);
                nearest_neighbor = length(anchor_idx);
                weight = lam - D;
                gamma = weight / sum(weight);
                
                y_anchor = X * W(:,anchor_idx) + w0(anchor_idx);
                y_predict = gamma * y_anchor';
                % SGD update
                err = max(0, 1-y*y_predict);
                loss = loss + err;
                
                if err > 0
                    W(:,anchor_idx) = W(:,anchor_idx) + learning_rate / (idx + t0) * y * repmat(gamma,p,1) .* repmat(X',1,nearest_neighbor);
                    w0(anchor_idx) = w0(anchor_idx) + learning_rate / (idx + t0) * y * gamma;

                    % update anchor points (SAPL)
                    s = 2 * LC * (repmat(X, nearest_neighbor, 1) - anchors(anchor_idx, :)).*repmat(weight, p, 1)';
                    base = -s * sum(weight.*y_anchor);
                    base = base + repmat(y_anchor',1,p).* s*sum(weight);
                    anchors(anchor_idx,:) = anchors(anchor_idx,:) + learning_rate_anchor / (idx + t0) * (y* base/(sum(weight).^2));
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

            min_nn = anchors_num;
            max_nn = 0;
            nn_test = 0;

            tic;
            for k=1:num_sample_test

                X = test_X(k,:);
                y = test_Y(k,:);

                [anchor_idx, D, lam] = Dynamic_KNN(anchors, X, LC);
                nearest_neighbor = length(anchor_idx);
                nn_test = nn_test + nearest_neighbor;

                if min_nn > nearest_neighbor
                    min_nn = nearest_neighbor;
                end

                if max_nn < nearest_neighbor
                    max_nn = nearest_neighbor;
                end

                weight = lam - D;
                gamma = weight / sum(weight);
                
                y_anchor = X * W(:,anchor_idx) + w0(anchor_idx);
                y_predict = gamma * y_anchor';
                err = max(0, 1-y_predict*y);
                loss = loss + err;

                if (y_predict>=0 && y==1) || (y_predict<0&&y==-1)
                    correct_num = correct_num + 1;
                end

            end
            toc;

            loss_fm_test(i,t) = loss / num_sample_test;
            fprintf('test loss:%.4f\t', loss_fm_test(i,t));
            accuracy_fm(i,t) = correct_num/num_sample_test;
            fprintf('\ttest accuracy:%.4f\t', accuracy_fm(i,t));

            min_nn_test(i, t) = min_nn;
            max_nn_test(i, t) = max_nn;
            nn_avg_test(i, t) = nn_test / num_sample_test;
            fprintf('\tnn avg:%.4f', nn_avg_test(i,t));

            fprintf('\n');

        end
        
        % toc;
    end
    
    % pack output
    % model
    model.w0 = w0;
    model.W = W;
    
    % metric
    metric.loss_train = loss_fm_train;
    metric.loss_test = loss_fm_test;
    metric.loss_accuracy = accuracy_fm;
    metric.max_nn_test = max_nn;
    metric.min_nn_test = min_nn;
    metric.nn_avg_test = nn_avg_test;

end