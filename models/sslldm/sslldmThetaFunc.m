function [theta] = sslldmThetaFunc(phi, psi, bias, phiMean, thetaMean, psiMean, biasMean, phiPrec, thetaPrec, psiPrec, biasPrec, theta0, options, data, rdata, dataN, numDoc, numWord, numRating, dim)
    phi = reshape(phi, [dim, numWord])';
    psi = reshape(psi, [dim, numRating])';
    theta = minFunc(@crossEntropyFunc,theta0, options);
    %theta = fminunc(@crossEntropyFunc,theta0, options);
    function [fval, fgrad] = crossEntropyFunc(theta)
        theta = reshape(theta, [dim, numDoc])';
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
        
        fgrad = thetaPrec*(theta - thetaMean);
        for d=1:numDoc
            q1 = zeros(1,dim);
            for i=1:numWord
                q1 = q1 + (Y(d,i)*dataN(d) - data(d,i))*phi(i,:);
            end
            fgrad(d,:) = fgrad(d,:) + q1;
        end
        fgrad = reshape(fgrad', [dim*numDoc, 1]);
    end
end