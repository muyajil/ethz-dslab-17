import argparse
from pydoc import locate
from scipy.misc import imsave

model = None
dataset = None

def predict_test_set(model, test_set, batch_size):
    image_index = 0
    for batch in test_set.batch_iter(batch_size):
        predicted = model.predict_batch(batch)
        for image in predicted:
            imsave('image_' + str(test_set._indices[image_index]) + '.jpg', image)

def run_model(run_mode, batch_size, epochs, split_ratio, image_dim):
    """Runs a model based on command line arguments
    """
    if run_mode == "debug":
        batch_iterator = dataset.batch_iter(batch_size, image_dim)
        for i in range(2):
            batch = next(batch_iterator)
    
    if run_mode == "test":
        train_set, test_set = dataset.split(split_ratio)
        for i in range(epochs):
            statistics = model.train_epoch(train_set, batch_size, image_dim, testset=test_set, test_period=200)
            print(statistics)
            model.save_model(epoch=i)
            print("Epoch " + str(i) + " finished.")
        model.save_model()
        predict_test_set(model, test_set, batch_size)
    
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
    parser.add_argument("split_ratio", metavar="split_ratio", type=float,
                        help="Ratio of data that should be in train set")
    parser.add_argument("img_width", metavar="img_width", type=int,
                        help="Width of image")
    parser.add_argument("img_height", metavar="img_height", type=int,
                        help="Width of image")
    parser.add_argument("img_channels", metavar="img_channels", type=int,
                        help="Width of image")
                    
    args = parser.parse_args()

    print(args)

    image_dim = (args.img_height, args.img_width, args.img_channels)

    dataset_module = locate("datasets." + args.dataset_name)
    dataset = getattr(dataset_module, "dataset")

    model_module = locate("models." + args.model_name)
    model = getattr(model_module, "model")

    model.set_up_model(image_dim)
    run_model(args.run_mode, args.batch_size, args.epochs, args.split_ratio, image_dim)
