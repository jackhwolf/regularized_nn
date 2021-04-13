Implementation of the regularized neural network architeture and objective function we discussed

I wrote a simple experiment that will 
- train model on training data
- test model against testing data and report
    - squared error
    - prediction accuracy
    - sparsity by layer of model
- graph model interpolation across domain
- save results to 

To run an experiment, run the command
`python3 experiment.py Files/Input/sample_input.yaml`
This will start running an experiment with the parameters defined in the `yaml` file. 

Results get stored to `Files/results.json`. 