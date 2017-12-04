import argparse
import os
import threading
from pydoc import locate
from models.res2pix import *
from utils import Dimension

model = None
dataset = None


def run_tensorboard(logdir, pythonpath):
    import os
    os.system(pythonpath + " -m tensorboard.main --logdir=" + logdir)
    return


def run_model(epochs, split_ratio):
    training_set, validation_set = dataset.split(split_ratio)
    with tf.Session() as sess:
        model.train(sess, training_set, epochs, validation_set=validation_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pass the arguments for the model run")
    parser.add_argument("dataset_name", metavar="dataset", type=str,
                        help="The name of the dataset that should be used")
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

    args = parser.parse_args()

    print(args)

    input_dimensions = Dimension(args.img_height, args.img_width, args.img_channels)

    dataset_module = locate("datasets." + args.dataset_name)
    dataset = getattr(dataset_module, "dataset")
    dataset_config = getattr(dataset_module, "config")
    dataset.initialize(dataset_config, args.base_path, input_dimensions, args.batch_size)

    t = threading.Thread(target=run_tensorboard, args=([args.log_dir, args.pythonpath]))
    t.start()
    
    model = Res2pix(config=Config(args.batch_size,
                                    input_dimensions,
                                    args.log_dir,
                                    pretrain_epochs=20,
                                    debug=True,
                                    gen_lambda=10,
                                    learning_rate=0.0005,
                                    stages=6,
                                    show_jpeg=True))
    run_model(args.epochs, args.split_ratio)
