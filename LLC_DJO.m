% reset rand seed
rng('default');

% load training data
[num_sample, p] = size(train_X);

% parameters

epoch = 10;
learning_rate = 1e5;
learning_rate_anchor = 100;%1000;
t0 = 1e4;
skip = 10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%skip_para=[1,10,100,1e3,1e4,1e5];
learning_rate_para=[1e4,1e5,1e6,1e7,1e8];
learning_rate_anchor_para = [1,10,100,1e3,1e4,1e5];
%learning_rate_anchor_para = [1100,1200,700,1300];
%diverse regularizatin
reg = 1;
skip1 = 0.01;
epsilon = 0.01; 
skip1_para = [1e-4,1e-3,1e-2,1e-1,1,10,1e2,1e3,1e4];
%epsilon_para = [1e-5,1e-4,1e-3,1e-2,1e-1,1,10,1e2,1e3,1e4];
%anchors_num_para =  [10, 20, 30, 40 ,50, 80, 100];
% locally linear anchor points
anchors_num = 50;

LC = 1;

iter_num = 5;%size(skip1_para,2); % size(learning_rate_anchor_para,2);
loss_JO_test = zeros(iter_num, epoch);
loss_JO_train = zeros(iter_num, epoch);
nn_test_JO = zeros(iter_num, epoch);
accuracy_JO = zeros(iter_num, epoch);

nn_train = zeros(iter_num, epoch);

para_set = size(skip1_para,2);
%para_set = 1;
for h = 1:para_set
   %learning_rate= learning_rate_para(h);
   %learning_rate_anchor = learning_rate_anchor_para(h);
   %anchors_num = anchors_num_para(h);
   skip1 = skip1_para(h);
    epsilon = epsilon_para(h);
    fprintf('This parameter is %f\n',skip1);    
for i=1:iter_num
    %skip = skip_para(i);
    %learning_rate_anchor = learning_rate_anchor_para(i);
    %skip1 = 0.00001;%skip1_para(i);
    %fprintf('This parameter is %f\n',skip1);
    
    num_nn = 0;
    minmum_K = 100;
    maximum_K = 0;
    
    b = zeros(1, anchors_num);
    W = zeros(p, anchors_num);
    loss_cumulative_JO = zeros(1, num_sample);                                     
    
    % initial anchor points via K-means
    %fprintf('Start K-means...\n');
    [~, anchors, ~, ~, ~] = litekmeans(train_X, anchors_num, 'Replicates', 1);
    %fprintf('K-means done..\n');
    
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
            if mod(j,1e10)==0
                toc;
                fprintf('%d iter(%d epoch)---processing %dth sample\n', i, t, j);
                fprintf('batch average value of K in KNN is %.2f\n', num_nn_batch/1e3);
                fprintf('overall average value of K in KNN is %.2f\n', num_nn/((t-1)*num_sample+j));
                
                num_nn_batch = 0;
                tic;
            end
            
            X = X_train(j,:);
            y = Y_train(j,:);
            
            % pick nearest anchor points (adaptive KNN)
            [anchor_idx, D, lam] = Dynamic_KNN(anchors, X, LC);
            nearest_neighbor = length(anchor_idx);
            weight = lam - D;
            gamma = weight / sum(weight);
            
            y_anchor = X * W(:,anchor_idx) + b(anchor_idx);
            y_predict = gamma * y_anchor';
            
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
            err = 1 - y * y_predict;
            
            % cumulative training hinge loss
            idx = (t-1)*num_sample + j;
            if idx == 1
                loss_cumulative_JO(idx) = max(0,err);
            else
                loss_cumulative_JO(idx) = (loss_cumulative_JO(idx-1) * (idx-1) + max(0,err))/idx;
            end
            
            % record loss epoch-wise
            loss_JO_train(i, t) = loss_cumulative_JO(idx);
            
            % sgd update
            if err > 0
                W(:,anchor_idx) = W(:,anchor_idx) + learning_rate / (idx + t0) * y * repmat(gamma,p,1) .* repmat(X',1,nearest_neighbor);
                b(anchor_idx) = b(anchor_idx) + learning_rate / (idx + t0) * y * gamma;
                
                % update anchor points (SAPL)
                s = 2 * LC * (repmat(X, nearest_neighbor, 1) - anchors(anchor_idx, :));
                base = -s * sum(weight.*y_anchor);
                base = base + repmat(y_anchor',1,p).* s*sum(weight);
                anchors(anchor_idx,:) = anchors(anchor_idx,:) + learning_rate_anchor / (idx + t0) * (y* base/(sum(weight).^2));
            end
            
            % regularization
            count = count - 1;
            if count <= 0
                W = W * (1 - skip/(idx + t0));
                switch reg
                    case 1
                        anchors_direct = normr(anchors - repmat(mean(anchors),size(anchors,1),1));
                        tmp = diag(pi*ones(size(anchors_direct,1),1))+acos(anchors_direct*anchors_direct');
                        tmp = sort(tmp);
                        anchors_score = tmp(1,:) + tmp(2,:);
                        %anchors_score = anchors_score.^(-1);
                        anchors_score = anchors_score/sum(anchors_score);
                        
                        %anchors_score = ones(size(W,2),1);
                        %W = W  - skip1/(idx + t0)*(2*W/sum(sum(W.^2))-2*(diag(anchors_score)*pinv(W))');
                       % W = W  - skip1/(idx + t0)*(2*W/sum(sum(W.^2))-2*(W*diag(anchors_score)));
                       W = W  - skip1/(idx + t0)*(2*W/sum(sum(W.^2))-2*(W*(W'*W+epsilon*eye(size(W,2)))^(-1)*diag(anchors_score)));
                    otherwise
                        
                end
                
                      
                count = skip;
            end
            
        end
        
        nn_train(i, t) = nn_epoch/num_sample;
        

       
    end
    
            % validate epoch-wise
        loss = 0.0;
        correct_num = 0;
        %fprintf('validating\n');
        %tic;
        [num_sample_test, ~] = size(test_X);
        nn_test = 0;
        
        for k=1:num_sample_test
            
            if mod(k,1e10)==0
                toc
                fprintf('%d epoch(validation)---processing %dth sample\n',i, k);
                tic
            end
            
            X = test_X(k,:);
            y = test_Y(k,:);
            
            [anchor_idx, D, lam] = Dynamic_KNN(anchors, X, LC);
            nearest_neighbor = length(anchor_idx);
            nn_test = nn_test + nearest_neighbor;
            weight = lam - D;
            
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
        loss_JO_test(i, t) = loss / num_sample_test;
        
        % record test accuracy epoch-wise
        accuracy_JO(i,t) = correct_num / num_sample_test;
        
        nn_test_JO(i,t) = nn_test /num_sample_test;
        
        %toc;
        %fprintf('validation done\n');    
end
    fprintf('final training loss %f std is %f\n',sum(loss_JO_train(:,size(loss_JO_train,2))/size(loss_JO_train,1)),std(loss_JO_train(:,size(loss_JO_train,2))));
    fprintf('final test loss %f std is %f\n',sum(loss_JO_test(:,size(loss_JO_test,2)))/size(loss_JO_test,1),std(loss_JO_test(:,size(loss_JO_test,2))));
    fprintf('final accuracy is %f std is %f\n',sum(accuracy_JO(:,size(accuracy_JO,2)))/size(accuracy_JO,1),std(accuracy_JO(:,size(accuracy_JO,2))));
    
end
%% plot cumulative learning curve
%plot(loss_cumulative_JO, 'DisplayName', 'LLC-SAPL');
%legend('-DynamicLegend');
%xlabel('Number of samples seen');
%ylabel('Hinge loss');
%grid on;


%% plot learning curve epoch-wise
%plot(loss_JO_train(1,:),'b--o', 'DisplayName', 'LLC-JO');
%legend('-DynamicLegend');
%xlabel('epoch');
%ylabel('Hinge loss');
%title('Cumulative Learning Curve')
%grid on;

%% plot nn epoch-wise
%plot(nn_train,'b--o', 'DisplayName', 'nn');
%legend('-DynamicLegend');
%xlabel('epoch');
%ylabel('nn');
%title('Average NN number Learning Curve')
%grid on;

%%
%x = [0.5,1,2,5,10,20,40, 50];
%y = [0.4266,0.4022,0.3730,0.3398,0.3411,0.3654,0.3874,0.4143];
%z = [19.8432, 14.068, 8.0765, 3.7899, 2.4170, 1.6845, 1.2125, 1.1047];
%a = [0,50];
%b = [0.3720,0.3720];
%plot(a,b,'k--','DisplayName', 'LLC-SAPL baseline');
%hold on
%plot(x,y,'b--o');
%title('sensity of lipschitz to noise ratio')
%hold on;
%grid on;
%ylabel('test hinge loss');
%yyaxis right
%plot(x,z,'r--o');
%xlabel('\mu');
%ylabel('average nn number');
%hold on;

