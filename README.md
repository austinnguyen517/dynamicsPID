# dynamicsPID
- Use of gaussian processes to represent objective space with respect to PID parameter space
- Implementation of bayesian optimization to find optimal PID parameters using expected improvement acquisition function
- Searching for optimal training parameters for ensemble neural network modeling dynamics of ionocraft 
- Graphs of predicted rollouts, training/testing loss of model with respect to epochs and loss gradients with respect to PID parameters

All files can be found in folder: PID

PID:
  Graphs: Visual representations of results from rollouts, PID parameter search, and model training. Contains all graphs throughout optimization and tuning process
  
  Out of date/Faulty implementation files:
  - DynamicsModel.py
  - ExecuteSim.py
  - ExpectedImprovement.py 
  - PIDSearch.py
  
  File Descriptions:
  - Ensemble NN: wrapper for implentation of ensemble neural network
  - ExecuteTrain: wrapper for training ensemble neural network or general network. Graphs model training/testing loss. Contains method to return initial conditions for bayesian optimization
  - GenNN: general neural network class for training and optimization. Used in ensemble neural network.
  - PID: class simulating PID regulators (proportion, integral, derivative). Used to regulate roll, pitch, yaw and respective rates
  - PID_BO_iono: uses bayesian optimization to find optimal PID parameters. Expected improvement with Opto library. Objective functions use trained ensemble neural network and PID class
  - PNNLoss: probabalistic loss of neural networks. Used to optimize ensemble/general neural network
  - Parse: parses data files from quadcopter to data frames workable with training models
  - TestRollouts: takes PID parameters and plots simulated rollouts based on train model to test stability
  - TrainedModels.txt: Various trained models used for bayesian optimization and rollout simulation
