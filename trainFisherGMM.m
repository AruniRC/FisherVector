function [m,c,p] = trainFisherGMM(sxy,numClusters,numSamples)
%% Trains GMM for Fisher Vector calculation
% numSamples controls how many samples from whole dataset are to be considered for clustering
% numClusters is the number of GMM clusters
% sxy is the entire dataset as p x n matrix, p - descriptor size = 66, 
%                                            n - number of data samples
% Returns means, covariances and priors of trained GMM

    
    %% Initializing with K-means
     S = sxy(:,1:numSamples); %reduced number of samples
     
     [initMeans, assignments] = vl_kmeans(S, numClusters,'Algorithm','Lloyd', ...
                                            'Initialization','plusplus');
                                                              
        initCovariances = zeros(66,numClusters);
        initPriors = zeros(1,numClusters);

        % Find the initial means, covariances and priors
        for i=1:numClusters
            data_k = S(:,assignments==i);
            initPriors(i) = size(data_k,2) / numClusters;

            if size(data_k,1) == 0 || size(data_k,2) == 0
                initCovariances(:,i) = diag(cov(S'));
            else
                initCovariances(:,i) = diag(cov(data_k'));
            end
        end
                                        
    
    %% GMM training on 66-dim PCA-SIFT-xy data
    
    disp('Training GMM . . .');
    data = single(S);
    [m, c, p] = vl_gmm(data, numClusters, ...
        'initialization','custom', ...
            'InitMeans',single(initMeans), ...
            'InitCovariances',single(initCovariances), ...
            'InitPriors',single(initPriors) );
    disp('Done.');
    
    disp('Saving GMM data to GMM-Fisher.mat');
    save('GMM-Fisher.mat','m','c','p','-7.3');
    
    disp('Returning means, covariances, priors of GMM clusters.');
    
end
