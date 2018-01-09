function [theta, phi, psi, bias] = lldm(phiPriorMean, thetaPriorMean, psiPriorMean, biasPriorMean, phiInit, thetaInit, psiInit, biasInit, phiPrec, thetaPrec, psiPrec, biasPrec, options, data, rdata, maxiter, dataNumber, impFeat)

dataN = sum(data,2);
savefile = strcat('modelsResult', num2str(dataNumber), '.mat');

numDocs = size(data, 1);
numWords = size(data, 2);
numRatings = size(rdata, 2);
ndim = size(phiPriorMean, 2);

psiInit = reshape(psiInit', [ndim*numRatings,1]);
phiInit = reshape(phiInit', [ndim*numWords,1]);
thetaInit = reshape(thetaInit', [ndim*numDocs,1]);

phi = rslldmPhiFunc(thetaInit, psiInit, biasInit, phiPriorMean, thetaPriorMean, psiPriorMean, biasPriorMean, phiPrec, thetaPrec, psiPrec, biasPrec, phiInit, options, data, rdata, dataN, numDocs, numWords, numRatings, ndim);
theta = rslldmThetaFunc(phi, psiInit, biasInit, phiPriorMean, thetaPriorMean, psiPriorMean, biasPriorMean, phiPrec, thetaPrec, psiPrec, biasPrec, thetaInit, options, data, rdata, dataN, numDocs, numWords, numRatings, ndim);
psi = rslldmPsiFunc(theta, phi, biasInit, phiPriorMean, thetaPriorMean, psiPriorMean, biasPriorMean, phiPrec, thetaPrec, psiPrec, biasPrec, psiInit, options, data, rdata, dataN, numDocs, numWords, numRatings, ndim);
bias = rslldmBiasFunc(theta, psi, phi, phiPriorMean, thetaPriorMean, psiPriorMean, biasPriorMean, phiPrec, thetaPrec, psiPrec, biasPrec, biasInit, options, data, rdata, dataN, numDocs, numWords, numRatings, ndim);
fprintf(1,'LLDM Iteration: %f\n', 1);
for i=2:maxiter
    phi = rslldmPhiFunc(theta, psi, bias, phiPriorMean, thetaPriorMean, psiPriorMean, biasPriorMean, phiPrec, thetaPrec, psiPrec, biasPrec, phi, options, data, rdata, dataN, numDocs, numWords, numRatings, ndim);
    theta = rslldmThetaFunc(phi, psi, bias, phiPriorMean, thetaPriorMean, psiPriorMean, biasPriorMean, phiPrec, thetaPrec, psiPrec, biasPrec, theta, options, data, rdata, dataN, numDocs, numWords, numRatings, ndim);
    psi = rslldmPsiFunc(theta, phi, bias, phiPriorMean, thetaPriorMean, psiPriorMean, biasPriorMean, phiPrec, thetaPrec, psiPrec, biasPrec, psi, options, data, rdata, dataN, numDocs, numWords, numRatings, ndim);
    bias = rslldmBiasFunc(theta, psi, phi, phiPriorMean, thetaPriorMean, psiPriorMean, biasPriorMean, phiPrec, thetaPrec, psiPrec, biasPrec, bias, options, data, rdata, dataN, numDocs, numWords, numRatings, ndim);
    
    wordRep = reshape(phi, [ndim, numWords])';
    docRep = reshape(theta, [ndim, numDocs])';
    ratingRep = reshape(psi, [ndim, numRatings])';
    maxiterVal = i;
    save(savefile, 'wordRep', 'docRep', 'ratingRep', 'bias', 'maxiterVal', 'impFeat');
    fprintf(1,'LLDM Iteration: %f\n', i);
    
end

phi = reshape(phi, [ndim, numWords])';
theta = reshape(theta, [ndim, numDocs])';
psi = reshape(psi, [ndim, numRatings])';

end
