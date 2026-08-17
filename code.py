import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time

countt=0 #global variable to always plot different test cases
plots=True #(if set to false, don't plot)

def euclideanDistance(x,y): #helper function for kNearestNeigh (slowest version)
    #convert types for negative number support
    x=x.astype(np.float32)
    y=y.astype(np.float32)
    return np.linalg.norm(x-y) #L2 norm

def manhattanDistance(x,y): #helper function for kNearestNeigh (slowest version)
    #convert types for negative number support
    x=x.astype(np.float32)
    y=y.astype(np.float32)
    return np.sum(np.abs(x-y)) #L1 norm

def cosineDistance(x,y): #helper function for kNearestNeigh (slowest version)
    #convert types for negative number support
    x=np.array(x,dtype=np.float32).reshape(-1) # convert to 1D array to avoid errors
    y=np.array(y,dtype=np.float32).reshape(-1)
    return 1 - (np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))) #convert to float to avoid errors

    #when vectors have the same direction, cosineDistance=0
    #when vectors are opposite, cosineDistance=2

def handleDistanceMetric(centroids,modifiedTest_set,typeDistance):
    if typeDistance=='m':

        predictions = []
        for i in range(len(modifiedTest_set)):
                distances = np.sum(np.abs(centroids - modifiedTest_set[i]),axis=1)  # L1 norm, axis=1 is columns (784 pixels), axis=0 is lines (60000 images)
                predictions.append(np.argmin(distances)) #append class with the smallest distance

    elif typeDistance == 'e':

        # Euclidean distance: ||A-B||^(2)=||A||^(2) +||B||^(2) - 2(A*B)

         centroids=np.array(centroids)
         centroid_norms = np.sum(centroids ** 2, axis=1)
         test_norms = np.sum(modifiedTest_set ** 2, axis=1)
         multiplication = 2 * np.dot(modifiedTest_set, centroids.T)  # Transpose to avoid error

         # [:,np.newaxis] represents each line (test image), [np.newaxis,:] represents each column (train image)
         distances = test_norms[:, np.newaxis] + centroid_norms[np.newaxis, :] - multiplication
         predictions=np.argmin(distances, axis=1) #append class with the smallest distance

    elif typeDistance == 'c':
        predictions=[]
        centroids = np.array(centroids)
        centroid_norms = np.linalg.norm(centroids, axis=1)
        for i in range(len(modifiedTest_set)):
            modifiedTest_setNorms =np.linalg.norm(modifiedTest_set[i])
            distances = 1 - (np.dot(centroids, modifiedTest_set[i]) / (centroid_norms*modifiedTest_setNorms))
            predictions.append(np.argmin(distances)) #append class with the smallest distance

    return predictions

def testAlgorithm(algorithm,train_set,test_set,k,typeDistance):
    global countt,plots
    start = time.time()

    if k is not None and typeDistance is not None:
        predictions=algorithm(train_set,test_set,k,typeDistance)
    elif k is not None:
        predictions=algorithm(train_set,test_set,k)
    else:
        predictions=algorithm(train_set,test_set,typeDistance)

    end = time.time()
    correct = 0
    true_labels=[] #used for confusion matrix
    for i in range(len(test_set)):
        true_labels.append(test_set[i]['label'])
        if predictions[i] == test_set[i]['label']:
            correct += 1
        if i<3 and countt<100 and plots:
           plt.imshow(test_set[countt]['image'], cmap='gray')
           plt.title(f"Predicted: {predictions[countt]}, Actual: {test_set[countt]['label']}")
           plt.show()
           countt+=1
    percentage = correct / len(test_set) * 100
    if plots:
        cm = confusion_matrix(true_labels, predictions)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    if k is not None:
         print(algorithm.__name__,"correct:", round(percentage,4), "% with k:", k,"Time taken:", round(end - start,2),"seconds, using distance:", typeDistance)
         if plots:
            plt.title(f"Confusion Matrix: {algorithm.__name__}, k:{k},typeDistance: {typeDistance}")
    else:
        print(algorithm.__name__, "correct:", round(percentage,4), "% Time taken:", round(end - start,2),"seconds, using distance:", typeDistance)
        if plots:
            plt.title(f"Confusion Matrix: {algorithm.__name__},typeDistance: {typeDistance}")
    if plots:
         plt.show()
    return percentage




#typeDistance: e for Euclidean distance, m for  Manhattan distance, c for cosine distance

def kNearestNeighSlow(train_set,test_set,k,typeDistance): #slower function, avoiding ready functions
    predictions=[]
    labels=train_set['label']
    for i in range (len(test_set)):
        distances=[]
        for j in range (len(train_set)):
             if typeDistance=='m':
                distances.append((manhattanDistance(train_set[j]['image'],test_set[i]['image']),labels[j]))
             elif typeDistance=='e':
                distances.append((euclideanDistance(train_set[j]['image'], test_set[i]['image']), labels[j]))
             elif typeDistance=='c':
                 distances.append((cosineDistance(train_set[j]['image'], test_set[i]['image']), labels[j]))
        distances.sort(key=lambda x: x[0])  #sort based on distances

        kLabels=[] #list that holds k labels with the smallest distance
        for j in range (k):
          kLabels.append(distances[j][1])
        max=0
        for j in range (len(kLabels)):
            count = 0
            for n in range (len(kLabels)):
                if kLabels[j]==kLabels[n]:
                    count+=1
            if count>max:
               max=count
               maxLabel=kLabels[j]
        predictions.append(maxLabel)
    return predictions

#typeDistance: e for Euclidean distance, m for  Manhattan distance, c for cosine distance

def kNearestNeighFast(train_set,test_set,k,typeDistance): #faster function for faster results

      #reshape converts array from (60000,28,28) to (60000,784) for faster results
      #float 32 to be able to handle negatives (for Euclidean distance)

    modifiedTrain_set=np.array(train_set['image']).reshape(len(train_set),-1).astype(np.float32)
    modifiedTest_set=np.array(test_set['image']).reshape(len(test_set),-1).astype(np.float32)
    labels=train_set['label']
    predictions=[]
    if typeDistance=='c':
        modifiedTrain_norm=np.linalg.norm(modifiedTrain_set,axis=1)
    for i in range (len(modifiedTest_set)):
        if typeDistance=='m':
            distances=np.sum(np.abs(modifiedTrain_set-modifiedTest_set[i]),axis=1) # L1 norm, axis=1 is columns (784 pixels), axis=0 is lines (60000 images)
        elif typeDistance=='e':
            distances = np.linalg.norm(modifiedTrain_set - modifiedTest_set[i],axis=1)  # L2 norm, axis=1 is columns (784 pixels), axis=0 is lines (60000 images)
        elif typeDistance=='c':
            distances = 1-(np.dot(modifiedTrain_set,modifiedTest_set[i]))/(modifiedTrain_norm*np.linalg.norm(modifiedTest_set[i]))
        k_indexes=np.argpartition(distances,k)[:k] #k nearest neighbors, np.argpartition to avoid sorting, np.argpartition puts smaller k as first indexes
        k_labels=labels[k_indexes] #take labels according to indexes
        counts=np.bincount(k_labels)  #bincount finds appearances of each neighbor
        most_common=counts.argmax() #argmax finds most common label
        predictions.append(most_common)
    return predictions


def kNearestNeighOptimized(train_set, test_set, k,typeDistance): #optimized function (when using Euclidean distance, or cosine distance), for fastest results

    # reshape converts array from (60000,28,28) to (60000,784) for faster results
    # float 32 to be able to handle negatives (for Euclidean distance)

    modifiedTrain_set = np.array(train_set['image']).reshape(len(train_set), -1).astype(np.float32)
    modifiedTest_set = np.array(test_set['image']).reshape(len(test_set), -1).astype(np.float32)
    labels = np.array(train_set['label'])

    if typeDistance=='e':

        #Euclidean distance: ||A-B||^(2)=||A||^(2) +||B||^(2) - 2(A*B)
        train_norms=np.sum(modifiedTrain_set**2,axis=1)
        test_norms=np.sum(modifiedTest_set**2,axis=1)
        multiplication=2*np.dot(modifiedTest_set,modifiedTrain_set.T) #Transpose to avoid error
        # [:,np.newaxis] represents each line (test image), [np.newaxis,:] represents each column (train image)
        distances=test_norms[:,np.newaxis]+train_norms[np.newaxis,:]-multiplication

    elif typeDistance=='c':

         train_norms=np.linalg.norm(modifiedTrain_set,axis=1)
         test_norms=np.linalg.norm(modifiedTest_set,axis=1)
         dot_products=modifiedTest_set @ modifiedTrain_set.T
         # [:,np.newaxis] represents each line (test image), [np.newaxis,:] represents each column (train image)
         distances=1-dot_products/(test_norms[:,np.newaxis]*train_norms[np.newaxis,:])

    predictions = []
    for i in range(len(modifiedTest_set)):
        k_indexes = np.argpartition(distances[i], k)[
            :k]  # k nearest neighbors, np.argpartition to avoid sorting, np.argpartition puts smaller k as first indexes
        k_labels = labels[k_indexes]  # take labels according to indexes
        counts = np.bincount(k_labels)  # bincount finds appearances of each neighbor
        most_common = counts.argmax()  # argmax finds most common label
        predictions.append(most_common)

    return predictions

#typeDistance: e for Euclidean distance, m for  Manhattan distance, c for cosine distance

def nearestCentroidSlow(train_set, test_set,typeDistance): #slower function, avoiding ready functions

    # reshape converts array from (60000,28,28) to (60000,784) for faster results
    # float 32 to be able to handle negatives (for Euclidean distance)

    modifiedTrain_set = np.array(train_set['image']).reshape(len(train_set), -1).astype(np.float32)
    modifiedTest_set = np.array(test_set['image']).reshape(len(test_set), -1).astype(np.float32)
    labels = np.array(train_set['label'])

    centroids=[] #list that holds centroid of each class
    for i in range (len(np.unique(labels))): #0,1,2...9
        sum=0
        count=0
        for j in range (len(modifiedTrain_set)):
            if labels[j]==i: #if the label is the same
              sum+=modifiedTrain_set[j]
              count+=1
        centroid=1/count*sum
        centroids.append(centroid)
    return handleDistanceMetric(centroids, modifiedTest_set, typeDistance)

#typeDistance: e for Euclidean distance, m for  Manhattan distance, c for cosine distance

def nearestCentroidOptimized(train_set, test_set,typeDistance): #optimized function for fastest results

    # reshape converts array from (60000,28,28) to (60000,784) for faster results
    # float 32 to be able to handle negatives (for Euclidean distance)

    modifiedTrain_set = np.array(train_set['image']).reshape(len(train_set), -1).astype(np.float32)
    modifiedTest_set = np.array(test_set['image']).reshape(len(test_set), -1).astype(np.float32)
    labels = np.array(train_set['label'])

    numClasses=len(np.unique(labels)) #number of unique classes

    mask=labels[:,np.newaxis]==np.arange(numClasses)  #boolean array mask[i,c], if i has label c, true
    # e.g. (labels=[0,1] ,numClasses=2, mask=[True, False], [False, True])

    centroids=mask.T @ modifiedTrain_set #(numClasses,num_sample) * (num_samples,num_features)

    counts=mask.sum(axis=0)[:,np.newaxis] #axis=0 for numClasses

    centroids=centroids/counts
    return handleDistanceMetric(centroids, modifiedTest_set, typeDistance)





dataset=load_dataset('ylecun/mnist')
dataset.set_format(type='numpy')

train_set=dataset['train']
test_set=dataset['test']


print("Find correctness percentage and time taken for each algorithm, using different k's and distance metrics")
print("e for Euclidean distance, m for  Manhattan distance, c for cosine distance\n")
print("Plotting 3 predictions for each algorithm:")
print("find optimal k for k-nn using Euclidean distance\n")

max=0
for k in range(1,10,2):
    percentage = testAlgorithm(kNearestNeighOptimized, train_set, test_set, k, 'e')
    if percentage > max:
        max = percentage
        best_k = k
print("\nOptimal k with fewest mismatches:",best_k,"with ",round(max,4),"% correct matches\n")
print("find optimal k for k-nn using cosine distance\n")

max=0
for k in range(1,10,2):
    percentage = testAlgorithm(kNearestNeighOptimized, train_set, test_set, k, 'c')
    if percentage > max:
        max = percentage
        best_k = k
print("\nOptimal k with fewest mismatches:",best_k,"with ",round(max,4),"% correct matches\n")
print("Testing nearest centroid classifier (both versions) for each distance metric:\n")

testAlgorithm(nearestCentroidOptimized, train_set, test_set,None,'m')
testAlgorithm(nearestCentroidSlow, train_set, test_set,None,'m')
testAlgorithm(nearestCentroidOptimized, train_set, test_set,None, 'e')
testAlgorithm(nearestCentroidSlow, train_set, test_set,None, 'e')
testAlgorithm(nearestCentroidOptimized, train_set, test_set,None, 'c')
testAlgorithm(nearestCentroidSlow, train_set, test_set,None, 'c')

train_set = train_set.select(range(6000))
test_set = test_set.select(range(1000))
print("\nShortening train set and test set (10 times smaller) (Manhattan distance takes more time)\n")
print("Using k-nn classifier:")
print("Using Manhattan distance:\n")
max=0
for k in range(1,10,2):

    percentage = testAlgorithm(kNearestNeighFast, train_set, test_set, k, 'm')

    if percentage > max:
        max = percentage
        best_k = k
print("\nOptimal k with fewest mismatches:",best_k,"with ",round(max,4),"% correct matches\n")


#to use the slower kNearestNeigh versions, shorten train and test set, for normal completion time
print("\nShortening train set and test set (10 times smaller) for slow version of k-nn\n")
train_set = train_set.select(range(600))
test_set = test_set.select(range(100))
max=0
print("Finding optimal k for k-nn using Euclidean distance (for slow version of algorithm)\n")
for k in range(1,10,2):
    percentage = testAlgorithm(kNearestNeighSlow, train_set, test_set, k, 'e')
    if percentage > max:
        max = percentage
        best_k = k
print("\nOptimal k with fewest mismatches:",best_k,"with ",round(max,4),"% correct matches\n")

print("Finding optimal k for k-nn using manhattan distance (for slow version of algorithm)\n")
max=0
for k in range(1,10,2):
    percentage = testAlgorithm(kNearestNeighSlow, train_set, test_set, k, 'm')
    if percentage > max:
        max = percentage
        best_k = k
print("\nOptimal k with fewest mismatches:",best_k,"with ",round(max,4),"% correct matches\n")

print("Finding optimal k for k-nn using cosine distance (for slow version of algorithm)\n")
max=0
for k in range(1,10,2):
    percentage = testAlgorithm(kNearestNeighSlow, train_set, test_set, k, 'c')
    if percentage > max:
        max = percentage
        best_k = k
print("\nOptimal k with fewest mismatches:",best_k,"with ",round(max,4),"% correct matches\n")














