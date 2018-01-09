function [] = models(datafile, startFoldNum, endFoldNum, useJointMutInf, priorAval, datatype, vecSpaceDim, maxiter, topN, psiPrec, numOfProcess)

    if ~isdeployed
        addpath(genpath('../../minFunc_2012'));
    end
    
    if ~isdeployed
        addpath(genpath('../../FEAST'));
    end
    
    if(useJointMutInf == 1)
        disp('Joint Mutual Information not supported');
    end
    
    data = [];
    docTerm = [];
    label = [];
    impFeature = [];
    load(datafile, 'docTerm', 'label', 'impFeature', 'wordRepPrior', 'docRepPrior', 'biasPrior', 'psiPrior');
    startFoldNum = str2double(startFoldNum);
    useJointMutInf = str2double(useJointMutInf);
    endFoldNum = str2double(endFoldNum);
    priorAval = str2double(priorAval);
    vecSpaceDim = str2double(vecSpaceDim);
    maxiter = str2double(maxiter);
    topN = str2double(topN);
    psiPrec = str2double(psiPrec);
    numOfProcess = str2double(numOfProcess);
    
    if strcmp(datatype,'nonSpones')
        data = docTerm;
    elseif strcmp(datatype,'spones') || strcmp(datatype,'sponesFreq')
        data = spones(docTerm);
    elseif strcmp(datatype,'aclPaper')
        [weightEachClass, indMutInfEachClass, weights, impFeat, indxImpf, indxImptf] = mutualInfo(docTerm, label, topN, topN, 50);
        data = spones(docTerm(:,indxImpf));
        impFeature = indxImpf;
    elseif strcmp(datatype,'impFeatureProvided')
        data = spones(docTerm(:,impFeature));
    elseif strcmp(datatype,'impFeature')
        if(useJointMutInf == 1)
            impFeature = feast('jmi',topN,full(docTerm),label);
        else
            impFeature = mutualInformation(docTerm, label, topN);
        end
        data = spones(docTerm(:,impFeature));
    elseif strcmp(datatype,'imdbImpFeatureProvided')
        docTerm = docTerm(~(label == 0), :);
        label = label(~(label == 0), :);
        data = spones(docTerm(:,impFeature));
    elseif strcmp(datatype,'imdbImpFeature')
        docTerm = docTerm(~(label == 0), :);
        label = label(~(label == 0), :);
        if(useJointMutInf == 1)
            impFeature = feast('jmi',topN,full(docTerm),label);
        else
            impFeature = mutualInformation(docTerm, label, topN);
        end
        data = spones(docTerm(:,impFeature));
    end

    myCluster = parcluster('local');
    myCluster.NumWorkers = numOfProcess;
    saveProfile(myCluster);   
    matlabpool open local;

    phiPrec = 1;
    thetaPrec = 1;
    psiPrec = psiPrec;
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
            elseif strcmp(datatype,'aclPaper')
                impFeat = impFeature;
            elseif strcmp(datatype,'sponesFreq')
                [weightEachClass, indMutInfEachClass, weights, impFeat, indxImpf, indxImptf] = mutualInfo(xtrain, lab, topN, topN, 50);
                impFeat = indxImpf;
            else
                [weightEachClass, indMutInfEachClass, weights, impFeat, indxImpf, indxImptf] = mutualInfo(xtrain, lab, topN, 20, 20);
                %impFeat = mutualInformation(xtrain, lab, topN);
            end
            if strcmp(datatype,'aclPaper')
	    	dat = full(xtrain);
            else
		xtrain = xtrain(:,impFeat);
                dat = full(xtrain);
	    end
        else
            dat = full(data);
            lab = label;
        end

        numWords = size(dat, 2);
        numDocs = size(dat, 1);
        
        order = unique(lab);
        numRatings = length(order);
        rdata = zeros(numDocs, numRatings);
        for i = 1:numRatings
            rdata(:,i) = (lab == order(i));
        end
        
        phiPriorMean = zeros(numWords, vecSpaceDim);
        thetaPriorMean = zeros(numDocs, vecSpaceDim);
        biasPriorMean = zeros(numWords, 1);
        psiPriorMean = zeros(numRatings, vecSpaceDim);

        if priorAval == 1
            phiPriorMean = wordRepPrior;
            biasPriorMean = biasPrior;
        end

        phiInit = rand(numWords, vecSpaceDim);
        thetaInit = rand(numDocs, vecSpaceDim);
        biasInit = rand(numWords, 1);
        psiInit = rand(numRatings, vecSpaceDim);
        
        if priorAval == 2
            thetaInit = docRepPrior;
            biasInit = biasPrior;
            phiInit = wordRepPrior;
            psiInit = psiPrior;
        end

        [theta, phi, psi, bias] = lldm(phiPriorMean, thetaPriorMean, psiPriorMean, biasPriorMean, phiInit, thetaInit, psiInit, biasInit, phiPrec, thetaPrec, psiPrec, biasPrec, options, dat, rdata, maxiter, x, impFeat);
    end
end
