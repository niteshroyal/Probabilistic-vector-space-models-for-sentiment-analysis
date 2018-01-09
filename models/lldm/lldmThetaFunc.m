function [theta] = lldmThetaFunc(phi, bias, phiMean, thetaMean, biasMean, phiPrec, thetaPrec, biasPrec, theta0, options, data, dataN, numDoc, numWord, dim)
    phi = reshape(phi, [dim, numWord])';
    theta = minFunc(@crossEntropyFunc,theta0, options);
    %theta = fminunc(@crossEntropyFunc,theta0, options);
    function [fval, fgrad] = crossEntropyFunc(theta)
        theta = reshape(theta, [dim, numDoc])';
        p1 = 0.5*thetaPrec*sumsqr(theta - thetaMean);
        p2 = 0.5*phiPrec*sumsqr(phi - phiMean);
        p3 = 0.5*biasPrec*sumsqr(bias - biasMean);
        Y = bsxfun(@plus, theta*phi', bias');
        Y = exp(Y);
        Y = bsxfun(@rdivide, Y, sum(Y,2));
        logY = log(Y);
        t1 = logY.*data;
        p4 = sum(t1(:));
        fval = p1 + p2 + p3 - p4;
        
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