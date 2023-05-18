import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
np.random.seed(0)
#SETUP


y = np.append(np.random.normal(4,1,1000) , np.random.normal(6,1,1000))
x = np.append(np.random.normal(4,1,1000) , np.random.normal(6,1,1000))


c1 = [3,1]
c2 = [8,4]

C = np.array([c1,c2])

data = np.array(list((zip(x,y))))
print(data.shape)
fig, ax = plt.subplots()

ax.scatter(x,y)

ax.scatter(c1[0], c1[1], marker= "*", color = "b")
ax.scatter(c2[0], c2[1], marker= "*", color = "r")


def dist(x1, xc):
    return np.sqrt((x1[0]-xc[0])**2 + (x1[1]-xc[1])**2)

#ALGORITHM
C1x = [c1[0]]
C1y = [c1[1]]
C2x = [c2[0]]
C2y = [c2[1]]

for epoch in range(10):

    cluster1 = []
    cluster2 = []
    for i in range(len(data)):
        
        distances = []
        for k in range(len(C)):
            ci = dist(data[i],C[k])
            distances.append(ci)
        
        classification = np.argmin(distances)
        if classification == 0:
            cluster1.append(data[i])
        else:
            cluster2.append(data[i])
        

    newC = []
    myclusters = [cluster1, cluster2]

    for k in range(len(C)):
        muk = 1/len(myclusters[k])   * np.sum(myclusters[k], axis=0)
        newC.append(muk)

    ax.scatter(newC[0][0],newC[0][1], marker= "^", color = "b")
    ax.scatter(newC[1][0],newC[1][1], marker= "^", color = "r")


    cluster1 = np.array(cluster1)
    cluster2 = np.array(cluster2)


    ax.scatter(cluster1[:,0],cluster1[:,1], color = [0,0,.8])
    ax.scatter(cluster2[:,0],cluster2[:,1], color = [.8,0,0])

    C1x.append(newC[0][0])
    C1y.append(newC[0][1])
    C2x.append(newC[1][0])
    C2y.append(newC[1][1])
    
    C = newC
    print(C)

  
    ax.plot(C1x, C1y, color = "b")
    ax.plot(C2x, C2y, color = "r")    
    plt.pause(1)

plt.show()