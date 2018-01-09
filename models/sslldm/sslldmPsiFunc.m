function [psi] = sslldmPsiFunc(theta, phi, bias, phiMean, thetaMean, psiMean, biasMean, phiPrec, thetaPrec, psiPrec, biasPrec, psi0, options, data, rdata, dataM, numDoc, numWord, numRating, dim)
    theta = reshape(theta, [dim, numDoc])';
    phi = reshape(phi, [dim, numWord])';
    psi = minFunc(@crossEntropyFunc,psi0, options);
    %phi = fminunc(@crossEntropyFunc,phi0, options);
    function [fval, fgrad] = crossEntropyFunc(psi)
        psi = reshape(psi, [dim, numRating])';
        p1 = 0.5*thetaPrec*sumsqr(theta - thetaMean);
        p2 = 0.5*phiPrec*sumsqr(phi - phiMean);
        p3 = 0.5*biasPrec*sumsqr(bias - biasMean);
        p4 = 0.5*psiPrec*sumsqr(psi - psiMean);
        Y = bsxfun(@plus, theta*phi', bias');
        Y = exp(Y);
        Y = bsxfun(@rdivide, Y, sum(Y,2));
        logY = log(Y);
        t1 = logY.*data;
        p5 = sum(t1(:));
        
        Z = phi*psi';
        Z = exp(Z);
        Z = bsxfun(@rdivide, Z, sum(Z,2));
        logZ = log(Z);
        t2 = logZ.*rdata;
        p6 = sum(t2(:));
        
        fval = p1 + p2 + p3 + p4 - p5 - p6;
        
        fgrad = psiPrec*(psi - psiMean);
        for j=1:numRating
            q1 = zeros(1,dim);
            for i=1:numWord
                q1 = q1 + (Z(i,j)*dataM(i) - rdata(i,j))*phi(i,:);
            end
            fgrad(j,:) = fgrad(j,:) + q1;
        end
        fgrad = reshape(fgrad', [dim*numRating, 1]);
    end
end