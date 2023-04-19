# AutoAda
This repository contains the result of my Master's thesis, _Adaptive Parameter Management in Machine Learning Systems_.
This repository has a big overlap to [AdaPM-PyTorch-apps](https://github.com/alexrenz/AdaPM-PyTorch-apps), to which I contributed as part of my work as a research assistant.
My main system contribution can be found in the ```auto_ada``` folder, ```ctr``` and ```gcn``` are the AdaPM use-cases ported to regular PyTorch. 


## Overview & Motivation
Training on AdaPM requires non-trivial modifications to existing codebases. 

 Models and training scripts need to be manually adapted  for AdaPM.
This requires in-depth knowledge of the model in question, in distributed computing and the underlying mechanisms of  AdaPM.
This is a substantial barrier for others to start using AdaPM.

My thesis therefore centered on removing those barriers. 
I created _AutoAda_ to automatically transform any PyTorch model, 
and I presented two ways to tackle the non-trivial task of intent signal generation.
The second, AutoAda automatic, **simulates future model parameter accesses efficiently** to derive intent signals (IS), 
which AdaPM depends upon to speed up distributed training.
### Results

| IS methods/ User requirements | Model Modification | Train script Modification | IS Generation                                 |
|-------------------------------|--------------------|---------------------------|-----------------------------------------------|
| manual IS                     | required           | substantial               | parameter access, access timing, Time keeping |
| AutoAda manual                | automatic          | minimal                   | parameter access only                         | 
| AutoAda automatic             |  automatic     | minimal                   | automatic                                     | 

Compared to the manual IS approach presented in [AdaPM-PyTorch-apps](https://github.com/alexrenz/AdaPM-PyTorch-apps),
my approaches simplify AdaPM use substantially, and any PyTorch model can be trained with AdaPM without additional user effort.

[//]: # (## Architecture)


[//]: # (## Performance)

