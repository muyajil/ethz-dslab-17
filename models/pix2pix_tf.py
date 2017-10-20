from models.abstract_model import ModelConfig
from models.abstract_tensorflow_model import AbstractTensorflowModel

class Pix2pix(AbstractTensorflowModel):
    
    def _new_model(self):
        pass
        
model = Pix2pix()