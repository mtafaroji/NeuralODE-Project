This AnyLogic model simulates a dynamic system with three stocks, which represent three state variables. The numerical values of these stocks are saved in three DataSets, and these DataSets are automatically exported as CSV files by an event in AnyLogic.
After running the simulation, the output CSV files should be placed in the data/raw/ folder.
The script load_anylogic_runs.py converts the CSV files into a single dataset file, dataset.pt, which is saved in the data/processed/ folder.
This dataset file can then be used by train_neural_ode.py to train the neural network. 