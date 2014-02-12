function [f] = trainFisherSIFT(img_cell)
%% Fisher Vector encoding on PCA-SIFT descriptors - calc  SIFT
%  at 5 scales for each image, scale factors of sqrt(2)
%  dimensionality reduction to 64 via PCA
% INPUT - cell array of cropped RGB LFW-deepfunneled images


       D = single(zeros(130,26261*length(img_cell)));
       G = single(zeros(130,26261));
       
  %% Forming 128-dim SIFT descriptor at dense patches over all images
    disp('Forming dense SIFT descriptor over all patches, images and 5 scales');
    disp('Working . . . ');
    
 
    for k=1:length(img_cell)
        k
        
        %convert to grayscale
        new_image=rgb2gray(img_cell{k}); 
        
        %resize to 160x125
        new_image = imresize(new_image,[160 125]);

        %convert to single precision
        single_image=im2single(new_image);

        % the [f, d] matrices represent the frames 
        % and descriptor matrices
        % d is 128xn
        
        % Over 5 scales
        for j = 1:5
              [f, d] = vl_dsift(single_image, 'step', 1, 'size', 6) ;
              for i=1:size(d,2)
                    d(129,i)=f(1,i);
                    d(130,i)=f(2,i);
              end

              G(:, (j-1)*size(d,2)+1 : j*size(d,2) ) = d;
              single_image = imresize(single_image, 1/sqrt(2));
        end
        D(:, (k-1)*26261+1 : k*26261 ) = G;
    end
    
    %% Saving to disk
    disp('SIFT descriptors for all images - done.');
    size(D);
    disp('Saving 128-dim data to disk . . . ');
    
    D = single(D);
    X = D(1:128,:)';
    save('D_SIFTxy.mat','D','-v7.3');
    save('D_SIFT.mat','X','-v7.3');
    disp('Done.');
    
    %% PCA-SIFT to reduce 128 to 64 dimensions
    % over all patches, all images and all scales
    disp('PCA to reduce 128 to 64-dim over all patches, all images and all scales . . .');
    [coeff, ~, ~] = pca(X,'NumComponents',64);
    disp('saving PCA coefficients to disk . . . ');
    save('PCA-SIFT_coeff.mat','coeff','-v7.3');
    disp('Done.');
    
    disp('Augmenting 64-dim PCA-SIFT with spatial coords');
    SIFT_128 = D(1:128,:);
    SIFT_xy = D(129:130,:);
    sift_64 = SIFT_128'*coeff; 
    sift_64 = sift_64';
    sxy = cat(1,sift_64,SIFT_xy); %Add spatial coords to 64-dim PCA-SIFT vector
    
    disp('Saving PCA-SIFT-xy data to disk . . . ');
    save('PCA-SIFT-data.mat','sxy','-v7.3');
    
    f = sxy;
    
    
end
