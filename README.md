# LUCID Neural Analysis
This analysis program was made so that various institutes such as CERN@School and the Institute for Research in Schools can use this program for analysing their data from Timepix or Medipix particle detectors.

[![N|Solid](http://cernatschool.web.cern.ch/sites/cernatschool.web.cern.ch/files/images/logos/IRIS_logo_white-backing.JPG)](http://www.researchinschools.org/)
[![N|Solid](https://cernatschool.web.cern.ch/sites/all/themes/cern/img/cern-logo-large.png)](https://cernatschool.web.cern.ch/)
[![N|Solid](https://lh3.googleusercontent.com/hIViPosdbSGUpLmPnP2WqL9EmvoVOXW7dy6nztmY5NZ9_u5lumMz4sQjjsBZ2QxjyZZCIPgucD2rhdL5uR7K0vLi09CEJYY=s688)](https://www.tensorflow.org/)

Model Folder
-------------
Contains the neural model used for classification.
The neural model can be viewed with Tensorboard with its respective accuracy and loss graphs.

LNA Test
--------
```
from lucid_neural.analysis import predict as classify

blob = [[0,0],[0,1],[1,0],[1,1],[0,2],[1,2],[0,3],[1,3]]

print(classify(blob))
>>> beta
```

Dependencies
------------
	- Tensorflow
	- LUCID Utils
	- Numpy