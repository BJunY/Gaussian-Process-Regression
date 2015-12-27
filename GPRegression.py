from numpy import *
import matplotlib.pyplot as plt

#define gaussian process kernal
def GPKernel(X, theta0, theta1, theta2):
    m = shape(X)[0]
    K1 = -0.5 * (transpose(kron(transpose(X), ones((m, 1)))) - kron(transpose(X), ones((m, 1))))**2
    K = theta0 *  exp(theta1*K1) + theta2 
    return (K1, K)

#compute the gradient respect to the hyperparameter
def computeGrad(Cn, grad, t):
    iCn = linalg.inv(Cn)
    gradient = -0.5*trace(dot(iCn, grad)) + 0.5*dot(dot(dot(transpose(t), iCn), grad), dot(iCn, t))
    return gradient

#approximate the hyperparameter, use the gaussian kernal and Gradient decent algorithom
def evaluateHyperPara(X , t, theta0, theta1, theta2, alpha, beta, iter_num):
    """
    use Gaussian kernal, Gradient decent 
    
    """    
    
    m = shape(X)[0] #输入的样本个数
    for i in range(iter_num):
        (K1, K) = GPKernel(X, theta0, theta1, theta2);
        Cn =  K + eye(m)/beta;
        grad0 = exp(theta1*K1);
        grad1 = theta0 * exp(theta1*K1)*K1;
        grad2 = ones((m, m));
        #grad3 = dot(X, transpose(X));
        gradb = eye(m);
        
        theta0 = theta0 - alpha*computeGrad(Cn, grad0, t);
        theta1 = theta1 - alpha*computeGrad(Cn, grad1, t);
        theta2 = theta2 - alpha*computeGrad(Cn, grad2, t);
       # theta3 = theta3 - alpha*computeGrad(Cn, grad3, t);
        beta   = beta - alpha*computeGrad(Cn, gradb, t);
        
    return (theta0, theta1, theta2, beta)

#predict the new point
def predict(Cn, t, testx, trainX, theta0, theta1, theta2, beta):
    K1 = -0.5 * ((trainX - testx)**2)
    K = theta0 *  exp(theta1*K1) + theta2 #+ theta3 * trainX * transpose(testx);
    a = dot(linalg.inv(Cn) , t)
    target = dot(transpose(K), a)
    return target

if __name__ ==  "__main__":
    #read data from the file
    X = genfromtxt(r"C:\Users\Bjy_PC\Desktop\input.txt")
    y = genfromtxt(r"C:\Users\Bjy_PC\Desktop\target.txt")
    
    #call the function to compute the hyperparameter
    (theta0, theta1, theta2, beta) = evaluateHyperPara(X, y, 1, 1, 1, 0.0001, 1, 300)#choose the initial parameter
    print '参数值为%f, %f, %f, %f'%(theta0, theta1, theta2, beta)
    
    #compute CN
    (K1, K) = GPKernel(X, theta0, theta1, theta2);
    m = shape(X)[0] #输入的样本个数
    Cn =  K + eye(m)/beta;
    
    #generate test data
    testX = linspace(-7, 7, 200)
    testY = ones(shape(testX))
    
    #predict the data
    for i in range(len(testX)):
        testY[i] = predict(Cn, y, testX[i], X, theta0, theta1, theta2 ,beta)
        print '>>已预测%d个点, 值为%f'%(i+1, testY[i])
        
    #plot the curve
    plt.plot(X, y, 'r*')
    plt.plot(testX, testY, 'g')
    plt.show()
    
    
    

    
        
    
