# Analog Printed Spiking Neuromorphic Circuit

This github repository is for the paper at DATE'24 - Analog Printed Spiking Neuromorphic Circuit

cite as
```
Analog Printed Spiking Neuromorphic Circuit
Pal, P.; Zhao, H.; Shatta, M,; Hefenbrock, M.; Mamaghani, S. B.; Nassif, S.; Beigl, M.; Tahoori, M. B.
2024 Design, Automation & Test in Europe Conference & Exhibition (DATE), IEEE, 2024

```


Usage of the code:

0. Modeling of the printed spiking g

In the folder `./simulation/` locate the data from SPICE simulation based on printed Processing Design Kit (pPDK). Different temporal input signals $\bold{x}_t$ are simulated and yielded the corresponding output signal $\bold{y}_t$. This part aims to build a machine learning based surroagate model of the printed spiking generator (pSG). Simply run the jupyter notebooks one by one 


~~~
1_read_cascade.ipynb
...
5_visualization.ipynb
~~~


1. Training of printed neural networks

After obtaining the machine learning based model of pSG, the whole circuit (including resistor crossbar for weighted-sum and pSG for nonlinearity) can be trained through

~~~
$ sh run_pSNN.sh
~~~

Alternatively, the experiments can be conducted by running command lines in `exp_pSNN.sh` separately, e.g.,

~~~
$ python3 exp_pSNN.py --DATASET 00 --SEED 0 --projectname pSNN
$ python3 exp_pSNN.py --DATASET 00 --SEED 1 --projectname pSNN
$ python3 exp_pSNN.py --DATASET 00 --SEED 2 --projectname pSNN
...
~~~

Analogous for baselines, the circuit can be trained through

~~~
$ sh run_SNN.sh
~~~

and 

~~~
$ sh run_pNN.sh
~~~


2. After training printed neural networks, the trained networks are in `./pSNN/model/`, the log files for training can be found in `./pSNN/log/`. If there is still files in `./pSNN/temp/`, you should run the corresponding command line to train the networks further. Note that, each training is limited to 48 hours, you can change this time limitation in `configuration.py`



3. Evaluation can be done by running the `Evaluation_pSNN.ipynb` in `./pSNN/` folder with
