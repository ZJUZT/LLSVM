function [ model, metric ] = svm( training, validation, pars )
%FM Summary of this function goes here
%   Detailed explanation goes here

    train_X = training.train_X;
    train_Y = training.train_Y;
    
    test_X = validation.test_X;
    test_Y = validation.test_Y;

    [num_sample, ~] = size(train_X);

    % parameters
    iter_num = pars.iter_num;
    learning_rate = pars.learning_rate;
    skip = pars.skip;
    count = skip;
    t0 = pars.t0;

    epoch = pars.epoch;

    loss_fm_test = zeros(iter_num, epoch);
    loss_fm_train = zeros(iter_num, epoch);
    accuracy_fm = zeros(iter_num, epoch);

    for i=1:iter_num

        % tic;

        w0 = pars.w0;
        W = pars.W;
%         V = pars.V;
        
        re_idx = randperm(num_sample);
        X_train = train_X(re_idx,:);
        Y_train = train_Y(re_idx);

        for t=1:epoch

            loss = 0;
            for j=1:num_sample

                X = X_train(j,:);
                y = Y_train(j,:);

                idx = (t-1)*num_sample + j;
                
                y_predict = W*X' + w0;

                % SGD update
                err = max(0, 1-y*y_predict);
                loss = loss + err;
                
                if err > 0
                    w0_ = learning_rate / (idx + t0) * (-y);
                    w0 = w0 - w0_;
                    W_ = learning_rate / (idx + t0) * (-y*X);
                    W = W - W_;
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

            tic;
            for k=1:num_sample_test

                X = test_X(k,:);
                y = test_Y(k,:);

                y_predict = W*X' + w0;
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
            fprintf('\ttest accuracy:%.4f', accuracy_fm(i,t));

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

end