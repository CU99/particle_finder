NN - Argon classification
-------------------------------

Reads npy files containing arrays of training and test data (SOAP descriptors). Network is trained and the accuracy on test data determined, and the networks makes predictions on the classification (liquid or solid) of each test particle.


Overall Argon classifer
-----------------------

Reads xyz file, creates SOAP descriptors, uses a pretrained network to make predictions on the classification of particles within this file, and creates colour-coded Mayavi plots.
