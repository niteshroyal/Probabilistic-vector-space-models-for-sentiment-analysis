function [phi] = sslldmPhiFunc(theta, psi, bias, phiMean, thetaMean, psiMean, biasMean, phiPrec, thetaPrec, psiPrec, biasPrec, phi0, options, data, rdata, dataN, dataM, numDoc, numWord, numRating, dim)
    theta = reshape(theta, [dim, numDoc])';
    psi = reshape(psi, [dim, numRating])';
    phi = minFunc(@crossEntropyFunc,phi0, options);
    %phi = fminunc(@crossEntropyFunc,phi0, options);
    function [fval, fgrad] = crossEntropyFunc(phi)
        phi = reshape(phi, [dim, numWord])';
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
        
        fgrad = phiPrec*(phi - phiMean);
        for j=1:numWord
            q1 = zeros(1,dim);
            for d=1:numDoc
                q1 = q1 + (Y(d,j)*dataN(d) - data(d,j))*theta(d,:);
            end
            for i=1:numRating
                q1 = q1 + (Z(j,i)*dataM(j) - rdata(j,i))*psi(i,:);
            end
            fgrad(j,:) = fgrad(j,:) + q1;
        end
        fgrad = reshape(fgrad', [dim*numWord, 1]);
    end
end