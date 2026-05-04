imds = imageDatastore('D:\yeni_D\elpv_edited_100_2\data\','IncludeSubfolders',true,'Labelsource','foldernames');

[imdsTrain,imdsValidation] =splitEachLabel(imds,0.9);
imageSize = [100 100 1];
pixelRange = [-4 4];

initialFilterSize = 3;
numInitialFilters = 16;
initialStride = 1;

numFilters = [16 32 64];
stackDepth = [4 3 2];

imageSize = [100 100 1];
numClasses = 2;

stackDepth = [3 4 23 3];
numFilters = [64 128 256 512];

lgraph2 = resnetLayers(imageSize,numClasses, ...
    StackDepth=stackDepth, ...
        NumFilters=numFilters)

        miniBatchSize = 128;
        learnRate = 0.001;
        valFrequency = 30;
        options = trainingOptions('adam', ...
            'InitialLearnRate',0.001, ...
                'MaxEpochs',500, ...
                    'Shuffle','every-epoch', ...
                        'ValidationData',imdsValidation, ...
                            'ValidationFrequency',20, ...
                                'MiniBatchSize',128,...
                                    'Verbose',false, ...
                                        'Plots','training-progress');
                                        [net,info] = trainNetwork(imdsTrain,lgraph2,options);
