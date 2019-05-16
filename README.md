# CNN_Hands

Convolutional Neural Network for drone controlling by hand gestures.

+ models on /outputs folder (generated weights and for CNN model)
+ dataset in /hands folder


+ AVision_Reporte.pdf 


## Instructions

For  windows use linux subsytem or any UNIX based OS

(https://www.youtube.com/watch?v=cVe-OR5neuc) - Windows Subsystem Tutorial

+ make sure you have installed Python3 in your _PATH_ (check the option)

https://www.python.org/downloads/

+ create a virtual evironment
```
$ python3 -m venv venv
```
+ Activate venv
```
$ source venv/bin/activate
```
+ Using Git subversion system (https://git-scm.com/downloads)
```
$ git clone https://github.com/RamonRomeroQro/Convolutional_Droning.git
```
+ install requirements
```
$ pip install -r req.txt
```
+ After connecting to the drone, run the program file
```
$ python3 droning.py

```

## Entrenar modelo CNN (opcional)
```
./train.sh
```

<!-- 
## probar modelo
+ ./run.sh [PATH-PRUEBA] -->
