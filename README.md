# LUCID Neural Analysis
This analysis program was made so that various institutes such as CERN@School and the Institute for Research in Schools can use this program for analysing their data from Timepix or Medipix particle detectors.

![CERN@School](http://cernatschool.web.cern.ch/sites/cernatschool.web.cern.ch/files/images/logos/IRIS_logo_white-backing.JPG)
![IRIS](https://cernatschool.web.cern.ch/sites/all/themes/cern/img/cern-logo-large.png)

Model Folder
-------------
The model folder contains the neural model used for classification. The neural model can be viewed with Tensorboard with its respective accuracy and loss graphs.
![Tensorflow](https://lh3.googleusercontent.com/hIViPosdbSGUpLmPnP2WqL9EmvoVOXW7dy6nztmY5NZ9_u5lumMz4sQjjsBZ2QxjyZZCIPgucD2rhdL5uR7K0vLi09CEJYY=s688)

LNA API
--------
```python
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

Research & Development
----------------------
[LUCID EPQ - Particle Analysis](https://www.scribd.com/document/356051031/Neural-Networks-for-Particle-Analysis-EPQ?secret_password=aBbTCmd88gkOCFan899j)
