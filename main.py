import argparse
import os
import threading
from pydoc import locate
from models.pix2pix import *
from utils import Dimension

model = None
dataset = None


def run_tensorboard(logdir, pythonpath):
    import os
    os.system(pythonpath + " -m tensorboard.main --logdir=" + logdir)
    return


def run_model(run_mode, epochs, split_ratio):
    """Runs a model based on command line arguments
    """
    if run_mode == "debug":
        debug_dataset = dataset.get_debug_dataset()
        debug_training, debug_validation = debug_dataset.split(0.66)
        model.train(debug_training, epochs=2, validation_set=debug_validation)
    
    if run_mode == "test":
        training_set, validation_set = dataset.split(split_ratio)
        model.train(training_set, epochs, validation_set=validation_set)
    
    if run_mode == "prod":
        model.train(dataset, epochs)

    # model.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pass the arguments for the model run")
    # parser.add_argument("model_name", metavar="model", type=str,
    #                     help="The name of the models that should be trained")
    parser.add_argument("dataset_name", metavar="dataset", type=str,
                        help="The name of the dataset that should be used")
    parser.add_argument("run_mode", metavar="run_mode", type=str, 
                        choices=["debug", "test", "prod"],
                        help="Choose one of the run modes: debug, test, prod")
    parser.add_argument("batch_size", metavar="batch_size", type=int,
                        help="Batch size")
    parser.add_argument("epochs", metavar="epochs", type=int,
                        help="Number of epochs to run")
    parser.add_argument("split_ratio", metavar="split_ratio", type=float,
                        help="Ratio of data that should be in train set")
    parser.add_argument("img_width", metavar="img_width", type=int,
                        help="Width of image")
    parser.add_argument("img_height", metavar="img_height", type=int,
                        help="Width of image")
    parser.add_argument("img_channels", metavar="img_channels", type=int,
                        help="Width of image")
    parser.add_argument("base_path", metavar="base_path", type=str,
                        help="Base path for the dataset")
    parser.add_argument("log_dir", metavar="log_dir", type=str,
                        help="Where to save the logs to")
    parser.add_argument("pythonpath", metavar="pythonpath", type=str,
                        help="Path to your python.exe")
    parser.add_argument("--restore_path", metavar="restore_path", type=str,
                        help="Restore path")

    args = parser.parse_args()

    print(args)

    input_dimensions = Dimension(args.img_height, args.img_width, args.img_channels)

    dataset_module = locate("datasets." + args.dataset_name)
    dataset = getattr(dataset_module, "dataset")
    dataset_config = getattr(dataset_module, "config")
    dataset.initialize(dataset_config, args.base_path, input_dimensions, args.batch_size)

    model_config = Config(args.batch_size, input_dimensions, args.log_dir, l1_lambda=10)
    if args.restore_path != '':
        model = Pix2pix(config=model_config, restore_path=args.restore_path)
    else:
        model = Pix2pix(config=model_config)

    t = threading.Thread(target=run_tensorboard, args=([args.log_dir, args.pythonpath]))
    t.start()

    run_model(args.run_mode, args.epochs, args.split_ratio)
