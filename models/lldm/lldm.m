function [theta, phi, bias] = lldm(phiPriorMean, thetaPriorMean, biasPriorMean, phiInit, thetaInit, biasInit, phiPrec, thetaPrec, biasPrec, options, data, maxiter, dataNumber, impFeat)

savefile = strcat('modelsResult', num2str(dataNumber), '.mat');
dataN = sum(data,2);

numDocs = size(data, 1);
numWords = size(data, 2);
ndim = size(phiPriorMean, 2);

phiInit = reshape(phiInit', [ndim*numWords,1]);
thetaInit = reshape(thetaInit', [ndim*numDocs,1]);

phi = lldmPhiFunc(thetaInit, biasInit, phiPriorMean, thetaPriorMean, biasPriorMean, phiPrec, thetaPrec, biasPrec, phiInit, options, data, dataN, numDocs, numWords, ndim);
theta = lldmThetaFunc(phi, biasInit, phiPriorMean, thetaPriorMean, biasPriorMean, phiPrec, thetaPrec, biasPrec, thetaInit, options, data, dataN, numDocs, numWords, ndim);
bias = lldmBiasFunc(theta, phi, phiPriorMean, thetaPriorMean, biasPriorMean, phiPrec, thetaPrec, biasPrec, biasInit, options, data, dataN, numDocs, numWords, ndim);
fprintf(1,'LLDM Iteration: %f\n', 1);
for i=2:maxiter
    phi = lldmPhiFunc(theta, bias, phiPriorMean, thetaPriorMean, biasPriorMean, phiPrec, thetaPrec, biasPrec, phi, options, data, dataN, numDocs, numWords, ndim);
    theta = lldmThetaFunc(phi, bias, phiPriorMean, thetaPriorMean, biasPriorMean, phiPrec, thetaPrec, biasPrec, theta, options, data, dataN, numDocs, numWords, ndim);
    bias = lldmBiasFunc(theta, phi, phiPriorMean, thetaPriorMean, biasPriorMean, phiPrec, thetaPrec, biasPrec, bias, options, data, dataN, numDocs, numWords, ndim);
    
    wordRep = reshape(phi, [ndim, numWords])';
    docRep = reshape(theta, [ndim, numDocs])';
    maxiterVal = i;
    save(savefile, 'wordRep', 'docRep', 'bias', 'maxiterVal', 'impFeat');
    fprintf(1,'LLDM Iteration: %f\n', i);
end

phi = reshape(phi, [ndim, numWords])';
theta = reshape(theta, [ndim, numDocs])';

end
