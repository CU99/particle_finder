NN - GeTe classification
-------------------------------

Reads npy files containing arrays of training and test data (SOAP and symmetry adapted functions). Network is trained and the accuracy on test data determined, and the networks makes predictions on the classification (alpha, beta, quenched) of each test particle.


Overall GeTe classifer
-----------------------

Reads xyz file, creates descriptors (combined SOAP and symmetry adapted functions), uses a pretrained network to make predictions on the classification of particles within this file, and creates colour-coded Mayavi plots.
