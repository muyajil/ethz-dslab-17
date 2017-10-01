class DatasetConfig:
    
    base_path = None
    base_name = None
    preprocess_pipeline = None
    num_datapoints = None
    load_function = None
    augmentation_multiplicator = None
    
    def __init__(self, base_path, base_name, preprocess_pipeline, num_datapoints,
                 load_function, augmentation_multiplicator):
        
        self.base_path = base_path
        self.base_name = base_name
        self.preprocess_pipeline = preprocess_pipeline
        self.load_function = load_function
        self.num_datapoints = num_datapoints
        self.augmentation_multiplicator = augmentation_multiplicator