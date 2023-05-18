import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
np.random.seed(0)
#SETUP

x = np.append(np.random.normal(4,1,1000) , np.random.normal(4,1,1000))
x= np.append(x, np.random.normal(10,1,1000))

y = np.append(np.random.normal(4,1,1000) , np.random.normal(12,1,1000))
y= np.append(y, np.random.normal(8,1,1000))

c1 = [0,1]
c2 = [4,10]
c3 = [7,1]
C = [c1,c2,c3]
print(C)
color = cm.rainbow(np.linspace(0, 1, len(C)))
data = list(zip(x,y))

fig, ax = plt.subplots()

ax.scatter(x,y)
ax.scatter(c1[0], c1[1], marker= "*", color = "b")
ax.scatter(c2[0], c2[1], marker= "*", color = "r")
ax.scatter(c3[0], c3[1], marker= "*", color = "g")

"""for cluster, c in zip(range(len(C)), color):
    ax.scatter(C[cluster][0], C[cluster][1], marker= "*", color=c)"""
   
def dist(x1, xc):
    return np.sqrt((x1[0]-xc[0])**2 + (x1[1]-xc[1])**2)

#ALGORITHM
C1x = [c1[0]]
C1y = [c1[1]]
C2x = [c2[0]]
C2y = [c2[1]]
C3x = [c3[0]]
C3y = [c3[1]]
for epoch in range(10):
    ax.clear()

    cluster1 = []
    cluster2 = []
    cluster3 = []
    for i in range(len(data)):
        
        distances = []
        for k in range(len(C)):
            ci = dist(data[i],C[k])
            distances.append(ci)
        
        classification = np.argmin(distances)
        if classification == 0:
            cluster1.append(data[i])
        elif classification == 1:
            cluster2.append(data[i])
        else:
            cluster3.append(data[i])
        
    newC = []
    myclusters = [cluster1, cluster2, cluster3]

    for k in range(len(C)):
        muk = 1/len(myclusters[k])   * np.sum(myclusters[k], axis=0)
        newC.append(muk)

    cluster1 = np.array(cluster1)
    cluster2 = np.array(cluster2)
    cluster3 = np.array(cluster3)


    ax.scatter(cluster1[:,0],cluster1[:,1], color = [0,0,.8])
    ax.scatter(cluster2[:,0],cluster2[:,1], color = [.8,0,0])
    ax.scatter(cluster3[:,0],cluster3[:,1], color = [0,.8,0])

    ax.scatter(newC[0][0],newC[0][1], marker= "^", color = "b")
    ax.scatter(newC[1][0],newC[1][1], marker= "^", color = "r")
    ax.scatter(newC[2][0],newC[2][1], marker= "^", color = "g")

    C1x.append(newC[0][0])
    C1y.append(newC[0][1])
    C2x.append(newC[1][0])
    C2y.append(newC[1][1])
    C3x.append(newC[2][0])
    C3y.append(newC[2][1])
    
    C = newC

    ax.plot(C1x, C1y, color = "b")
    ax.plot(C2x, C2y, color = "r")    
    ax.plot(C3x, C3y, color = "g")  
    
    plt.pause(3)

plt.show()