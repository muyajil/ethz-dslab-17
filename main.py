import argparse
from pydoc import locate

model = None
dataset = None

def run_model():
    """Runs a model based on command line arguments
    """
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pass the arguments for the model run")
    parser.add_argument("model_name", metavar="model", type=str, 
                    help="The name of the models that should be trained")
    parser.add_argument("dataset_name", metavar="dataset", type=str,
                    help="The name of the dataset that should be used")
    parser.add_argument("run_mode", metavar="run_mode", type=str, 
                    choices=["debug", "test", "prod"],
                    help="Choose one of the run modes: debug, test, prod")
    args = parser.parse_args()
    
    dataset = locate("datasets." + args.dataset_name + ".dataset")
    model = locate("models." + args.model_name + ".model")