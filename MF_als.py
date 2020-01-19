import numpy as np



def loss(R ,P, Q, K, alpha=0.001, lambd=0.02):
    #print R.shape,P.shape,Q.shape
    #print R
    #print P
    #print Q
    tempQ = Q.T
    loss = 0
    for i in range(len(R)):
        for j in range(len(R[0])):
            if R[i,j]>0:##no value is continue
                #print "p[i,:]",P[i,:]
                #print "Q[:,j]",tempQ[:,j]
                loss +=pow((R[i,j]-np.dot(P[i,:],tempQ[:,j])),2)#(observe-predict)^2
                for k in range(K):
                    loss+=(lambd/2)*(pow(P[i][k],2)+pow(tempQ[k][j],2))
    return loss

def als(R,P,Q,K,alpha=0.001,lambd=0.02):
    shape=(K,K)
    earray=np.ones(shape)
    E=np.mat(earray)
    lenP=len(P)
    lenQ=len(Q)
    steps=100
    for step in range(steps):
        lossvalue = 123
        for i in range(lenP):
            # print i
            # print Q.shape,(Q.T).shape
            # print Q
            M1 = np.dot((Q.T), Q)
            M1 = lambd * E + M1
            M1_1 = np.mat(M1).I
            QT = Q.T
            Ri = np.mat(R[i, :]).T
            Pi = np.dot(np.dot(M1_1, QT), Ri)
            P[i, :] = Pi[:, 0].T

        for j in range(lenQ):
            M1 = np.dot((P.T), P)
            M1 = lambd * E + M1
            M1_1 = np.mat(M1).I
            QT = P.T
            Ri = np.mat(R[:, j]).T
            Qi = np.dot(np.dot(M1_1, QT), Ri)
            Q[j, :] = Qi[:, 0].T
            lossvalue = loss(R, P, Q, K)
        if lossvalue < 0.01:
            break
        print("als,iteration:", step, "loss:", lossvalue)
    return P, Q



if __name__ == "__main__":
    R = [
     [5,3,0,1],
     [4,0,0,1],
     [1,1,0,5],
     [1,0,0,4],
     [0,1,5,4],
    ]##(5,4,), k=2
    R=np.array(R)
    N=len(R)
    M=len(R[0])
    K=2
    P=np.random.rand(N,K)
    Q=np.random.rand(M,K)
    nP,nQ=als(R,P,Q,K,0.001,0.02)
    print(R)
    print(np.dot(nP,nQ.T))
