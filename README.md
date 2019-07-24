# dynamicsPID
- Use of gaussian processes to represent objective space with respect to PID parameter space
- Implementation of bayesian optimization to find optimal PID parameters using expected improvement acquisition function
- Searching for optimal training parameters for ensemble neural network modeling dynamics of ionocraft 
- Graphs of predicted rollouts, training/testing loss of model with respect to epochs and loss gradients with respect to PID parameters

All files can be found in folder: PID

## PID:
  
  Graphs: Visual representations of results from rollouts, PID parameter search, and model training. Contains all graphs throughout optimization and tuning process
  
  ## File Descriptions:
  
  ### Wrappers/Fundamental Research Files:
  - SimBOModelCycle: wrapper to implement cycle between simulation data generation, model training, and BO to better represent system dynamics and find optimal PID parameters
  
  ### Bayesian Optimization:
  - PID_BO_iono: uses bayesian optimization to find optimal PID parameters. Expected improvement with Opto library. Objective functions use trained ensemble neural network and PID class
  
  ### Models:
  - Ensemble NN: wrapper for implentation of ensemble neural network
  - GenNN: general neural network class for training and optimization. Used in ensemble neural network.
  
  ### Model Training:
  - ExecuteTrain: wrapper for training ensemble neural network or general network. Graphs model training/testing loss. Contains method to return initial conditions for bayesian optimization
  
  ### Individual Files for External Usage
  - PID: class simulating PID regulators (proportion, integral, derivative). Used to regulate roll, pitch, yaw and respective rates
  - PIDPolicy: generates a policy given PID parameters 
  - PNNLoss: probabalistic loss of neural networks. Used to optimize ensemble/general neural network
  - Parse: parses data files from quadcopter to data frames workable with training models
  - kMeansData: Implements kMeans clustering on data points from quadcopter. Equally samples from each cluster to optimize model training
  
  ### Testing:
  - CrazyFlieSim: true simulation of quadcopter dynamics 
  - ExecuteSim: execute simulation to test fitness of PID parameters and policy
  - TrainedModels.txt: Various trained models used for bayesian optimization and rollout simulation
