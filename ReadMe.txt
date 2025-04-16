We used a main training loop (see "LearningAgent.py") to integrate the OpenAI Gym with
E3AC algorithm. We then trained our agent to verify the performance.

----------------------------------------------------------------------------

1. To run our Leaning Agent code, please ensure you fulfill all the package requirements mentioned in
Report1 (OpenAI Gym Report) and Report 2 (Neural Network Report).


Table I. Package Requirements in Report 1

Package   	Version	    Usage
python	                3.7.3	    /
numpy	                1.21.5	    /
scikit-fuzzy	0.4.2	    Fuzzy reward
matplotlib	3.5.3	    /
xlrd	                2.0.1	    Read xls files
pandas	                1.3.5	    Read xls files
CoppeliaSim	4.0.0 Edu     Simulation
Sim	                /	    RemoteAPI functions


Table II. Package Requirements in Report 2
Package	    Version	Package	               Version
python	    3.7.3	                matplotlib	3.5.3
numpy	    1.21.5	                pandas	                1.3.5
Tensorflow  1.14.0	                gym	                0.19.0

----------------------------------------------------------------------------

2. Explanations to the files under "OpenAI Gym Codes Group 3" directory


"E3AC_Model" file: well trained network parameters of E3AC in scene3.

"Training Data" file:  all the training results we presented in our report.

"iiwa_scene3.ttt": the robotic scenarios we established in CoppeliaSim.

"Learning Agent Report.pdf":  our report that contains refelections, modifications on previous methods, and explanations to main codes. 

"LearningAgent.py":  our codes for the main training loop.

"Neural_Network_E3AC.py": our codes for the proposed E3AC algorithm.

"OpenAIGym.py": our codes for the environment.

"ou_noise": we downloaded it online, it is a code for random OU noise, which is used in E3AC.

"remoteApi.dll": supporting plugin (for Windows system) for communications between CoppeliaSim and python

"remoteApi.so": supporting plugin (for Ubantu system) for communications between CoppeliaSim and python

"sim.py" and "simConst.py": supporting libraries provided by CoppeliaSim containing various remoteAPI functions

**** "remoteApi.dll, remoteApi.so, sim.py, and simConst.py" must be put under the directory of our codes ****

----------------------------------------------------------------------------

3. Instructions — How to run our Learning Agent:

*** We established 3 scenes with different complexity***
*** In this assignment, we take Scene 3 (the most complex one) as the training environment***

Step 1: Ensure you have installed CoppeliaSim, and fulfill all the package requirements.
Step 2: Open "iiwa_scene3.ttt" file and run it. Click "ok" if there are any pop-up windows.
Step 3: Run "OpenAIGym.py" in python. 

***Note: CoppeliaSim must be run before OpenAIGym.py***
***Note the code is set to run testing episodes. Set "ON_TRAIN = True" (line 12) to run training episodes***

