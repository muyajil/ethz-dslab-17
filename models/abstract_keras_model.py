import keras
import os
import time

from models.abstract_model import AbstractModel, ModelConfig


class AbstractKerasModel(AbstractModel):
    """Abstract keras model class
  
    Each keras model needs to derive from this class.
    """

    def train(self, training_set, epochs, validation_set=None):
        """Fits the model parameters to the dataset.
           Only one epoch.

        Args:
            training_set: Instance of AbstractDataset
            epochs: Number of epochs to train
            validation_set: Data on which to evaluate the model.
                         
        Returns:
            Nothing
        """
        print("Starting training..")
        steps_per_epoch = training_set.get_size() / self._config.batch_size
        print("Calling fit_generator with steps_per_epoch=" + str(steps_per_epoch))
        print("log_dir: " + self._config.log_dir)
        tfb_callback = keras.callbacks.TensorBoard(log_dir=self._config.log_dir,
                                                   histogram_freq=1,
                                                   batch_size=self._config.batch_size,
                                                   write_graph=True,
                                                   write_grads=True,
                                                   write_images=True,
                                                   write_batch_performance=True)
        
        mckp_callback = keras.callbacks.ModelCheckpoint(str(os.path.join(self._config.log_dir, 'model.ckpt')),
                                                        monitor='val_loss',
                                                        verbose=0,
                                                        save_best_only=False,
                                                        save_weights_only=False,
                                                        mode='auto',
                                                        period=1)

        return self._model.fit_generator(training_set.batch_iter(),
                                         steps_per_epoch,
                                         epochs,
                                         validation_data=validation_set.get_as_numpy_array() if validation_set is not None else None,
                                         validation_steps=validation_set.get_size() if validation_set is not None else None,
                                         shuffle=False,
                                         callbacks=[tfb_callback, mckp_callback])

    def validate(self, validation_set):
        """Validates the model on the provided dataset.
        
        Args:
            validation_set: Instance of AbstractDataset class
            
        Returns:
            Some score to measure the performance of the model.
        """
        return self._model.evaluate_generator(validation_set.batch_iter(self._config.batch_size),
                                              validation_set.get_size())

    def predict_batch(self, batch):
        """Returns a encoded then decoded datapoint

        Args:
            batch: A numpy array of datapoints
        """
        return self._model.predict_on_batch(batch)

    def _new_model(self):
        """Creates a new model.

        Returns:
            An instance of the model
        """
        raise NotImplementedError("Method not implemented.")


    def _restore_model(self, restore_path):
        """Restores a model stored at restore_path
        
        Args:
            restore_path: Path where the model is stored.
            
        Returns:
            An instance of the model.
        """
        
        return keras.models.load_model(restore_path)
        
        
    def _print_model_summary(self):
        """ Prints the model structure and other information.
        """
        self._model.summary()
    
