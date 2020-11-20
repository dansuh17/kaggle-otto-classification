import tensorflow as tf


class Predict:
    def __init__(self):
        pass

    def predict(self):
        pass


class PredictionModel:
    # Ensembles XGBoost and neural network models
    
    def __init__(self, xgb_model, nn_model):
        self.xgb_model = xgb_model
        self.nn_model = nn_model
        
    def predict(self, dataset):
        xgb_pred = self.xgb_model.predict_proba(dataset)
        
        dataset_tensor = tf.convert_to_tensor(dataset, dtype=tf.float32)
        print(dataset_tensor)
        nn_pred = self.nn_model.predict(dataset_tensor)
        
        print(xgb_pred.shape)
        print(nn_pred.shape)
        return (xgb_pred + nn_pred) / 2
