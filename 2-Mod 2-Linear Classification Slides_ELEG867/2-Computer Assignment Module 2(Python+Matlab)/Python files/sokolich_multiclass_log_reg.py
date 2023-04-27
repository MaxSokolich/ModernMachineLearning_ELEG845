
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import time 
np.random.seed(0)

#############################################################################
# Preparing the MNIST dataset for logistic regression
train_digits = sio.loadmat('/Users/bizzarohd/Desktop/ModernMachineLearning_ELEG845/2-Mod 2-Linear Classification Slides_ELEG867/2-Computer Assignment Module 2(Python+Matlab)/Python files/data_minst.mat')
# The dataset has two compoents: train_data and train_labels
# Train_data has 5000 digits images, and the dimension of each image is 28 * 28 (784 * 1)
# Train_lables has 5000 labels, 
train_data = train_digits['train_feats']
# There are totally 5000 images and the dimension of each image is 784 * 1
num_train_images = train_data.shape[0]
dim_images = train_data.shape[1]

train_bias = np.ones([num_train_images,1])
data_x = np.concatenate((train_data, train_bias), axis=1)

# Generating the labels for the one-verus-all logistic regression
labels = train_digits['train_labels']
data_y = np.zeros([num_train_images,10])
for i in range(num_train_images):
	data_y[i,labels[i][0]-1] = 1


#print(data_y.shape)

#############################################################################
# The logistic function
def logistic(x,beta):
    y_pred = np.matmul(x,beta)  #no use of bias term as indicated in assignment instructions
    logistic_prob = 1/(1 + np.exp(-y_pred))
    return logistic_prob

#############################################################################
# Here you create you own LOGISTIC REGRESSION algorith for the logistic regression model
# Hint: please check the slides of the logistic regression 
#############################################################################

def logistic_regression(beta, lr, x_batch, y_batch,lambda1):
	start = time.time()
	
	#FOR LOOP APPROACH
	#this could work...
	'''DNNL_list = []
	for j in range(len(beta)):
		current_DNNL = 0
		for i in range(len(x_batch)):
			current_DNNL +=  x_batch[i,j] * (y_batch[i] - logistic(x_batch[i,:-1] , beta[:-1])) + lambda1 * beta[j] 
	
		DNNL_list.append(-current_DNNL)
		
	
	DNNL_list = np.array(DNNL_list)

	beta_next = beta - learning_rate * DNNL_list
	print("beta_next shape = ", beta_next.shape)'''
	##############################################################
	"""#MATRIX APPROACH
	DNNL_list = []
	for j in range(len(beta)):
		c_DNNL = np.sum(x_batch[:,j] * (y_batch - logistic(x_batch, beta)))  +  lambda1 *beta[j]
		DNNL_list.append(-c_DNNL)

	DNNL_list = np.array(DNNL_list)
		
	beta_next = beta - learning_rate * DNNL_list
	print("beta_next shape = ", beta_next.shape)"""

	#different approach
	#cost = -np.sum(np.dot(y_batch , np.log(logistic(x_batch[:,:-1] , beta[:-1]))) + np.dot((1-y_batch) , np.log(1 - logistic(x_batch[:,:-1] , beta[:-1]))))
	def sigmoid(z):
		return 1 / (1 + np.exp(-z))
	
	y_pred = np.matmul(x_batch[:,:-1],beta[:-1].T) + beta[784]
	p = sigmoid(y_pred)
	
	cost = -np.sum(y_batch * np.log(p) + (1-y_batch)* np.log(1-p))
	regularization = (lambda1/2) * np.sum(beta**2)
	cost += regularization

	dcost =   -np.sum(x_batch.T * (y_batch - p), axis = 1) + lambda1 * beta 
	beta_next = beta - learning_rate * dcost
	

	return cost, beta_next

#############################################################################
 #409.761
def classifcation_ratio(beta,x_batch,y_batch):
	ratio = 0
	logistic_prob = logistic(x_batch,beta)
	clssifcation_result = logistic_prob > 0.5
	true_result = y_batch == 1
	comparison = np.logical_and(clssifcation_result,true_result)
	ratio = float(np.sum(comparison))/float(np.sum(true_result))
	return ratio 


#############################################################################
#                       Main Function
#############################################################################

# Hyper-parameters
training_epochs = 100
learning_rate = 0.0005          # The optimization initial learning rate
lambda1 = 1						# The regularization parameter
cost = 0
FINAL = []
for i in range(0,20): #loop through lambda
	lambda1 = i
	classification_list = []
	for j in range(10):
		current_label = data_y[:,0]
		print("current label (1-10)", current_label.shape)
		beta = np.random.randn(dim_images + 1)
		for epoch in range(training_epochs):
			cost, beta_next = logistic_regression(beta,learning_rate,data_x,current_label, lambda1)
			ratio = classifcation_ratio(beta,data_x,current_label)
			
			print('Class %d Epoch %3d, cost %.3f, the classification accuracy is %.2f%%' % (i+1, epoch+1,cost,ratio*100))
			beta = beta_next
		classification_list.append(ratio*100)
	FINAL.append(classification_list)

#############################################################################
# Visualizing five images of the MNIST dataset
fig, _axs = plt.subplots(nrows=1, ncols=5)
axs = _axs.flatten()

for i in range(5):
	axs[i].grid(False)
	axs[i].set_xticks([])
	axs[i].set_yticks([])
	image = data_x[i*10,:784]
	image = np.reshape(image,(28,28))
	aa = axs[i].imshow(image,cmap=plt.get_cmap("gray"))

fig.tight_layout()


#MY DISPLAY
fig, ax = plt.subplots()

for i in range(len(FINAL)): #should be 20 lists for each lambda
	ax.plot(FINAL[i], label = "lambda = {}, average_class_acc = {}".format(i,np.mean(FINAL[i])))
	ax.set_ylim([80,100])



ax.set_ylabel("classification accuracy")
ax.set_xlabel("ylabel ( digits 1-1)")
ax.legend()
plt.show()

