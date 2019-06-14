#define a model to get the state change given a state and an action
#initialize random param_values of PID inputs into the execute function
    #revise the code so that the return of the PID_BO_iono is all of the objective values
        #and the next set of PID parameters
#decide whether the objective values of the currently evaluated PID inputs was best so far
    #if it is, then set that as our new best
    #otherwise, just carry on with our next inputs
#use the pid parameters we had previously to get a new action. With the current state and action, return a new state


#fix the objective function to have some initialized generalnn
#then, using the passed PID parameters x (in PID_BO_iono), go through a few cycles (decide this) of passing in actions/states into the general nn and analyze what kind of states we get
    #calculate our loss based on our outputted states
#make a new OptTask with ^ that objective function and the parameters passed similar to PID_BO_iono
#set the stop criteria to about 50 (or whatever)
#define a dotmap with verbosity, acq_func, and model
#make a statement opt = opto.BO(parameters = p, task = task, stopCriteria = stop)
#call opt.optimize
#logs = opt.get_logs()
#call logs.get_parameters() to get the optimal pid parameters
#to get the number of evaluations call log.get_n_evals()
