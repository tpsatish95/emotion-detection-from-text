from sklearn import cluster, datasets

X_iris = [[0,1,3,5,2], [231,43,45,12,23], [23,131,33,35,32],[546,16,36,54,62],[40,14,36,56,26]]

print (X_iris)

k_means = cluster.KMeans(n_clusters=5)
k_means.fit(X_iris) 
print(k_means.labels_)
