import os
import time
import tensorflow as tf

from models.abstract_model import AbstractModel, ModelConfig

class AbstractTensorflowModel(AbstractModel):
    """Abstract tensorflow model class
  
    Each tensorflow model needs to derive from this class.
    """
