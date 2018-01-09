function [] = models(datafile, startFoldNum, endFoldNum, useJointMutInf, priorAval, datatype, vecSpaceDim, maxiter, topN, numOfProcess, wPrior, dPrior)

    if ~isdeployed
        addpath(genpath('../../minFunc_2012'));
    end
    
    if ~isdeployed
        addpath(genpath('../../FEAST'));
    end
    
    if(useJointMutInf == 1)
        disp('Joint Mutual Information not supported');
    end

    label = [];
    data = [];
    docTerm = [];
    cocurMatrixReal = [];
    cocurMatrixInt = [];
    impFeature = [];
    load(datafile, 'docTerm', 'label', 'cocurMatrixReal', 'cocurMatrixInt', 'impFeature', 'wordRepPrior', 'biasPrior', 'docRepPrior');
    endFoldNum = str2double(endFoldNum);
    wPrior = str2double(wPrior);
    dPrior = str2double(dPrior);
    startFoldNum = str2double(startFoldNum);
    priorAval = str2double(priorAval);
    vecSpaceDim = str2double(vecSpaceDim);
    maxiter = str2double(maxiter);
    topN = str2double(topN);
    useJointMutInf = str2double(useJointMutInf);
    numOfProcess = str2double(numOfProcess);
    
    if strcmp(datatype,'nonSpones')
        data = docTerm;
    elseif strcmp(datatype,'spones') || strcmp(datatype,'sponesFreq')
        data = spones(docTerm);
    elseif strcmp(datatype,'impFeatureProvided')
        data = spones(docTerm(:,impFeature));
    elseif strcmp(datatype,'impFeature')
        if(useJointMutInf == 1)
            impFeature = feast('jmi',topN,full(docTerm),label);
        else
            impFeature = mutualInformation(docTerm, label, topN);
        end
        data = spones(docTerm(:,impFeature));
    elseif strcmp(datatype,'coocurReal')
        data = cocurMatrixReal(impFeature,impFeature);
    elseif strcmp(datatype,'coocurInt')
        data = cocurMatrixInt(impFeature,impFeature);
    elseif strcmp(datatype,'coocurSpones')
        data = spones(cocurMatrixInt(impFeature,impFeature));
    end

    myCluster = parcluster('local');
    myCluster.NumWorkers = numOfProcess;  
    saveProfile(myCluster);   
    matlabpool open local;

    phiPrec = 1;
    thetaPrec = 1;
    biasPrec = 1;

    options = [];
    options.useMex = 1;
    options.MaxIter = 10;
    options.Method = 'lbfgs';
    options.Diagnostics = 'on';
    options.Display = 'iter';
    options.DerivativeCheck = 'off';
    options.UseParallel = 1;

    %options = optimoptions('fminunc','Algorithm', 'quasi-newton', 'Display', 'iter-detailed', 'HessUpdate', 'bfgs', 'MaxIter', 100, 'Diagnostics', 'on', 'DerivativeCheck', 'on', 'GradObj','on');
    
    len = size(data, 1);
    parfor x=startFoldNum:endFoldNum
        impFeat = impFeature;
        if endFoldNum>-1
            idxtrain = ones(len,1);
            idxtest = zeros(len,1);
            for i = 1:len
                if mod(i,10) == x
                    idxtest(i) = 1;
                    idxtrain(i) = 0;
                end
            end
            xtrain = data(logical(idxtrain),:);
            lab = label(logical(idxtrain),:);
            if(useJointMutInf == 1)
                impFeat = feast('jmi', topN, full(xtrain), lab);
            elseif strcmp(datatype,'sponesFreq')
                [weightEachClass, indMutInfEachClass, weights, impFeat, indxImpf, indxImptf] = mutualInfo(xtrain, lab, topN, topN, 50);
                impFeat = indxImpf;
            else
                [weightEachClass, indMutInfEachClass, weights, impFeat, indxImpf, indxImptf] = mutualInfo(xtrain, lab, topN, 20, 20);
                %impFeat = mutualInformation(xtrain, lab, topN);
            end
            xtrain = xtrain(:,impFeat);
            dat = full(xtrain);
        else
            dat = full(data);
        end

        numWords = size(dat, 2);
        numDocs = size(dat, 1);
        phiPriorMean = zeros(numWords, vecSpaceDim);
        thetaPriorMean = zeros(numDocs, vecSpaceDim);
        biasPriorMean = zeros(numWords, 1);

        if priorAval == 1
            %thetaPriorMean = docRepPrior;
            biasPriorMean = biasPrior;
            phiPriorMean = wordRepPrior;
        end

        phiInit = rand(numWords, vecSpaceDim);
        thetaInit = rand(numDocs, vecSpaceDim);
        biasInit = rand(numWords, 1);

        if priorAval == 2
            thetaInit = docRepPrior;
            biasInit = biasPrior;
            phiInit = wordRepPrior;
        end

        lbldm(phiPriorMean, thetaPriorMean, biasPriorMean, phiInit, thetaInit, biasInit, phiPrec, thetaPrec, biasPrec, options, dat, maxiter, x, impFeat, wPrior, dPrior);
    end
end
