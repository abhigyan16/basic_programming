knn = KNeighborsClassifier(n_neighbors = 1) 
  
knn.fit(X_train, y_train) 
pred = knn.predict(X_test) 
  
print('WITH K = 1') 
print('\n') 
print(confusion_matrix(y_test, pred)) 
print('\n') 
print(classification_report(y_test, pred)) 
  
  
# NOW WITH K = 15 
knn = KNeighborsClassifier(n_neighbors = 15) 
  
knn.fit(X_train, y_train) 
pred = knn.predict(X_test) 
  
print('WITH K = 15') 
print('\n') 
print(confusion_matrix(y_test, pred)) 
print('\n') 
print(classification_report(y_test, pred)) 
