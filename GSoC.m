%MATLAB Script to detect upper human bodies from Big Bang Theory images
%using Support Vector Machines.

%The training data is a set of images with bounding boxes of the upper bodies.
%Positive training examples are image patches extracted at the annotated locations. 
%A negative training example can be any image patch that does not significantly overlap
%with the annotated upper bodies. 

%Obtaining the training and validation data:
[trD, trLb, valD, valLb, trRegs, valRegs]   = Utils.getPosAndRandomNeg();

%Training Data
pos_x = trD(:,1:size(trLb(trLb==1)));
pos_y = trLb(1:size(trLb(trLb==1)))';

neg_x = trD(:,size(trLb(trLb==1))+1:size(trD,2));
neg_y = trLb(size(trLb(trLb==1))+1:size(trD,2))';

x_train = [pos_x neg_x];
y_train = [pos_y neg_y]';

%Validation Data
pos_x_val = valD(:,1:size(valLb(valLb==1)));
pos_y_val = valLb(1:size(valLb(valLb==1)))';

neg_x_val = valD(:,size(valLb(valLb==1))+1:size(valD,2));
neg_y_val = valLb(size(valLb(valLb==1))+1:size(valD,2))';
x_val = [pos_x_val neg_x_val];
y_val = [pos_y_val neg_y_val]';

%train_svm function returns the optimized weights and the loss.
[weights,loss] = train_svm(x_train,y_train);

plot(loss);

%Calculating training accuracy (Obtained: 99.18%)
[accuracy,~] = calc_accuracy(weights,x_train,y_train);
disp(accuracy);

%Calculating validation accuracy (Obtained: 96.89%)
[accuracy,~] = calc_accuracy(weights,x_val,y_val);
disp(accuracy);


function [weights,loss] = train_svm(x,y)
    
    loss = zeros(2000); %To plot the loss
    eta0 = 1;
    eta1 = 100;
    
    %Number of classes : currently, k = 2 
    %k=1 signifies presence of an upper body.
    %k=0 signifies absence of an upper body.
    k = size(unique(y),1); 
    
    %dimension and number of training samples
    [dim,num] = size(x);
    
    %Weight matrix of size (dimension x number of classes).
    w = zeros(dim,k);
    
    c=10;
    
    %Mapper from class names to integral class names.
    m=containers.Map([1,-1],[1,2]);

    %Stochastic Gradient Descent for SVM.
    for epoch=1:2000
        
        %Updating the learning rate.
        eta = eta0/(eta1+epoch);
        
        %Shuffling the indices of training data.
        randindex = randperm(num);
        totalloss=0;
        
        %Updating weights matrix.
        for i=1:num
            
            %Finding Y_hat first
            index=randindex(i);
            x_i=x(:,index);
            y_i=m(y(index));
            temp_w=w;
            temp_w(:,y_i)=(-1*inf);
            [~,y_hat]=max(temp_w'*x_i);
            l = max((w(:,y_hat)'*x_i-w(:,y_i)'*x_i+1),0);

            %Update rules for weight
            for j=1:k
                if j==y_i
                    if l>0
                        der_y_i=(w(:,y_i))./num - c.*(x_i);
                    else
                        der_y_i = (w(:,y_i))./num;
                    end            
                    w(:,j) = w(:,j) - (eta*der_y_i);
                elseif j==y_hat
                    if l>0
                        der_y_hat=(w(:,y_hat))./num + c.*(x_i);
                    else
                        der_y_hat =(w(:,y_hat))./num;
                    end            
                    w(:,j) = w(:,j) - (eta*der_y_hat);
                else
                    w(:,j) = w(:,j) - (eta*(w(:,j))./num); 

                end
            end
            
            %Multiclass Hinge Loss
            l = max((w(:,y_hat)'*x_i-w(:,y_i)'*x_i+1),0); 
            totalloss = totalloss + (sum(vecnorm(w).^2))/(2*num) + c*l;

        end
        loss(epoch) = totalloss;
        disp(totalloss);
    end
    weights = w;
end

%Function to calculate the accuracy given the weights, data and its labels.
function [accuracy,error] = calc_accuracy(weights,x,y)

    [~,y_hat_ind] = max(weights'*x);
    [num,~] = size(y);
    m=containers.Map([1,-1],[1,2]);
    y_hat_ind=y_hat_ind';
    y_actual =zeros(num,1);
    for i=1:num
        y_actual(i)=m(y(i));

    end
    accuracy = sum(y_hat_ind == y_actual)/num;
    error = sum(y_hat_ind ~= y_actual)/num;
end