# active_learning
Plan for research:
1) Get labeled training dataset, with good and bad photos
2) Run cnn iterations + svm to classify images.
  2.1) As a constant, try to see how this performs with completely labeled dataset
  2.2) Preprocess all of our (labeled) data into .npy file, called bottleneck_features.npy,
       and we also save a corresponding .npy file called bottleneck_labels.npy which
       stores the corresponding labels for the features.
  2.3) Using these bottleneck_features, and bottleneck_labels, we then train a svm
       on the features.
3)  
