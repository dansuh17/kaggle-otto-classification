import os
import numpy as np
import pandas as pd
import xgboost
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import log_loss
import tensorflow as tf
from tensorflow import keras
from prediction_model import PredictionModel
import dankag as dk


IN_DIR = '../input/otto-group-product-classification-challenge'
COL_TARGET = 'target'
COL_ID = 'id'


config = {
    'test_file': os.path.join(IN_DIR, 'test.csv'),
    'train_file': os.path.join(IN_DIR, 'train.csv'),
    'sample_submission_file': os.path.join(IN_DIR, 'sampleSubmission.csv'),
    'submission_file': 'submission.csv',
    'col_target': 'target',
    'col_id': 'id',
    'batch_size': 256,
}

train_data_read_step = dk.CSVDataReadStep(
        data_file=dk.StateVal('train_file'), 
        data_key='train_data', 
        target_key='train_target', 
        target_col=dk.StateVal('col_target'), 
        name='train_read')

test_data_read_step = dk.CSVDataReadStep(
        data_file=dk.StateVal('test_file'), 
        data_key='test_data', 
        name='test_read')

# feature transformation pipeline
feature_scaler = StandardScaler()

preprocessor = ColumnTransformer(transformers=[
        ('feature_transform', feature_scaler, slice(1, 94)),
        ('remove_id', 'drop', [COL_ID]),
], remainder='passthrough')

sklearn_preprocess_step = dk.SklearnTrainTestPreprocessorStep(
        train_data=dk.StateVal('train_data'), 
        test_data=dk.StateVal('test_data'), 
        preprocessor=preprocessor)

target_preprocess_step = dk.SklearnPreprocessorStep(
        data=dk.StateVal('train_target'), 
        preprocessor=LabelEncoder(), 
        preprocessor_key='target_preprocessor',
        data_out_key='train_target', 
        fit=True)


class SplitStep(dk.Step):
    def __init__(self, data, target, test_size: float, out_keys: list, name='split_step'):
        super().__init__(name)
        self.data = data
        self.target = target
        self.test_size = 0.2
        self.out_keys = out_keys

    def run(self, state) -> dict:
        data = self.stateval_or(self.data, state)
        target = self.stateval_or(self.target, state)

        x_train_key, x_val_key, y_train_key, y_val_key = self.out_keys

        x_train, x_val, y_train, y_val = train_test_split(
            data, target, test_size=self.test_size, random_state=9123)

        state[x_train_key] = x_train
        state[x_val_key] = x_val
        state[y_train_key] = y_train
        state[y_val_key] = y_val
        return state

# split the dataset
split_step = SplitStep(
        data=dk.StateVal('train_data'), 
        target=dk.StateVal('train_target'), 
        test_size=0.2, 
        out_keys=('train_data', 'val_data', 'train_target', 'val_target'))


class AggregateStep(dk.Step):
    def __init__(self, dataset_key: str, data, target, name='aggregate'):
        super().__init__(name)
        self.dataset_key = dataset_key
        self.data = data
        self.target = target

    def run(self, state) -> dict:
        state[self.dataset_key] = (
            self.stateval_or(self.data, state), 
            self.stateval_or(self.target, state)
        )
        return state

train_aggregate_step = AggregateStep('train_dataset', dk.StateVal('train_data'), dk.StateVal('train_target'))
val_aggregate_step = AggregateStep('val_dataset', dk.StateVal('val_data'), dk.StateVal('val_target'))

tf_dataset_train_step = dk.BatchedTFDatasetConversionStep(
        data=dk.StateVal('train_data'),
        target=dk.StateVal('train_target'),
        batch_size=dk.StateVal('batch_size'),
        dataset_key='tf_train_dataset')

tf_dataset_val_step = dk.BatchedTFDatasetConversionStep(
        data=dk.StateVal('val_data'),
        target=dk.StateVal('val_target'),
        batch_size=dk.StateVal('batch_size'),
        dataset_key='tf_val_dataset')


class OttoModel(dk.Model):
    def __init__(self, batch_size: int, num_classes=9):
        self.xgb_model = dk.XgbModel()
        
        initializer = tf.keras.initializers.HeNormal()

        nn_model = tf.keras.Sequential([
            tf.keras.Input(shape=(93, )),
            
            tf.keras.layers.Dense(128, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dropout(0.75),
            tf.keras.layers.Dense(20, activation='relu', kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dropout(0.75),
            tf.keras.layers.Dense(num_classes, activation='softmax'),
        ])

        self.nn_model = dk.TensorflowModel(nn_model, 
                optimizer=keras.optimizers.Adam(learning_rate=2e-4),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'],
                epochs=500)

        self.batch_size = batch_size

    def fit(self, train_dataset, val_datasets: list):
        tf_train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset).batch(self.batch_size)
        tf_val_datasets = [
            tf.data.Dataset.from_tensor_slices(vd).batch(self.batch_size) 
            for vd in val_datasets
        ]
        self.nn_model.fit(tf_train_dataset, tf_val_datasets)

        self.xgb_model.fit(train_dataset, val_datasets)

    def predict(self, dataset):
        dataset_tensor = tf.convert_to_tensor(dataset, dtype=tf.float32)
        return (self.xgb_model.predict(dataset) + self.nn_model.predict(dataset_tensor)) / 2

create_model_step = dk.CreateModelStep(
        OttoModel, model_key='model', batch_size=dk.StateVal('batch_size'))

model_train_step = dk.ModelTrainStep(
        model=dk.StateVal('model'), 
        train_dataset=dk.StateVal('train_dataset'),
        val_datasets=dk.StateVal('val_dataset'))

predict_step = dk.PredictStep(
        model=dk.StateVal('model'), 
        data=dk.StateVal('test_data'), 
        prediction_key='out_data')

output_step = dk.CSVOutputStep(
        file_name='submission.csv', out_data=dk.StateVal('out_data'))

steps = [
    train_data_read_step,
    test_data_read_step,

    # preprocess data
    sklearn_preprocess_step,
    target_preprocess_step,

    # split the dataset
    split_step,

    # to TF dataset
    # tf_dataset_train_step,
    # tf_dataset_val_step,
    train_aggregate_step,
    val_aggregate_step,

    # create model
    create_model_step,

    # train
    model_train_step,

    # predict using test data
    predict_step,

    # export results to output file
    output_step,
]


def train_calib(train_data, targets):
    model = XGBClassifier(
        learning_rate=0.05, 
        gamma=0.03,
        n_estimators=520,
        # n_estimators=1,  # debug purpose
        max_depth=8, 
        min_child_weight=1.5,
        colsample_bytree=0.8,
        subsample=0.8, 
        nthread=4, 
        verbosity=2,
        objective='multi:softprob')
    
    calib_model = CalibratedClassifierCV(model, cv=5, method='isotonic')
    calib_model.fit(train_data, targets)
    return calib_model


def score_model(model, data, targets):
    # test on validation set
    prds = model.predict_proba(data)
    return log_loss(targets, prds)

if __name__ == '__main__':
    # create a submission file
    # make_output(PredictionModel(xgb_model, model), X_test, ids=test_data_raw.id)
    dk.run(config, steps=steps)
