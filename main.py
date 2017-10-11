import argparse
from pydoc import locate
from datasets.dataset import Dataset

model = None
dataset = None

def run_model(run_mode, batch_size, epochs):
    """Runs a model based on command line arguments
    """
    if run_mode == "debug":
        batch_iterator = dataset.batch_iter(batch_size)
        for i in range(2):
            batch = next(batch_iterator)
    
    if run_mode == "test":
        train_set, test_set = dataset.split(0.9)
        for i in range(epochs):
            statistics = model.train_epoch(train_set, batch_size, testset=test_set, test_period=1)
    
    if run_mode == "prod":
        for i in range(epochs):
            statistics = model.train_epoch(dataset, batch_size)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pass the arguments for the model run")
    parser.add_argument("model_name", metavar="model", type=str, 
                    help="The name of the models that should be trained")
    parser.add_argument("dataset_name", metavar="dataset", type=str,
                    help="The name of the dataset that should be used")
    parser.add_argument("run_mode", metavar="run_mode", type=str, 
                    choices=["debug", "test", "prod"],
                    help="Choose one of the run modes: debug, test, prod")
    
    parser.add_argument("batch_size", metavar="batch_size", type=int,
                    help="Batch size")
    parser.add_argument("epochs", metavar="epochs", type=int,
                    help="Number of epochs to run")
                    
    args = parser.parse_args()
    
    dataset_module = locate("datasets." + args.dataset_name)
    dataset = getattr(dataset_module, "dataset")
    
    model_module = locate("models." + args.model_name)
    model = getattr(model_module, "model")
    model.set_up_model(dataset.get_data_dimension())
    run_model(args.run_mode, args.batch_size, args.epochs)