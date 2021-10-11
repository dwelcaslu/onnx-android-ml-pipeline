'''
ONNX Pipeline lib.
'''
import os, sys
# from pathlib import Path
# FILEPATH = str(Path(os.path.abspath(__file__)))
# PROJ_DIR = str(Path(os.path.abspath(__file__)).resolve().parents[(len(FILEPATH.split("\\")) - [i for i, x in enumerate(FILEPATH.split("\\")) if x == 'Butler'][0] - 2)])
# sys.path.append(os.path.join(PROJ_DIR, Path('RCA-AI/rca/')))
# sys.path.append(os.path.join(PROJ_DIR, Path('RCA-AI/src/')))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import tensorflow as tf
import onnxruntime as rt
from skl2onnx import convert_sklearn, update_registered_converter
# from skl2onnx.algebra.onnx_ops import OnnxMul, OnnxSin, OnnxSqrt, OnnxSub, OnnxPow, OnnxConcat, OnnxWhere, OnnxGreater, OnnxLess, OnnxOr, OnnxEqual
from skl2onnx.common.data_types import BooleanTensorType, FloatTensorType, Int64TensorType, StringTensorType, DoubleTensorType
# from skl2onnx.common.data_types import guess_numpy_type
# from skl2onnx.common._registration import register_shape_calculator


def convert_initial_types_schema(data, drop=None):
    inputs = []
    for k, v in zip(data.columns, data.dtypes):
        if drop is not None and k in drop:
            continue
        if v in ('int64', 'int32'):
            t = Int64TensorType([None, 1])
        elif v in ('float64', 'float32'):
            t = FloatTensorType([None, 1])
        else:
            t = StringTensorType([None, 1])
        inputs.append((str(k), t))
    return inputs

def convert_input_schema(data):
    inputs = {}
    for k, v in zip(data.columns, data.dtypes):
        if v in ('int64', 'int32'):
            inputs[str(k)] = data[k].values.astype(np.int64).reshape(-1, 1)
        elif v in ('float64', 'float32'):
            inputs[str(k)] = data[k].values.astype(np.float32).reshape(-1, 1)
        else:
            inputs[str(k)] = data[k].values.astype(str).reshape(-1, 1)
    return inputs

def convert_input_types(data, input_types):
    for k, v in input_types.items():
        if v in ('int64', 'int32', 'int'):
            data[k] = data[k].values.astype(int)
        elif v in ('float64', 'float32', 'float'):
            data[k] = data[k].values.astype(float)
        elif v in ('string', 'str'):
            data[k] = data[k].values.astype(str)
    return data


class ONNXConverter():
    def __init__(self, preprocessor=None, estimator=None):
        # Class inputs:
        if isinstance(preprocessor, Pipeline):
            self.preprocessor = preprocessor
        elif isinstance(preprocessor, ColumnTransformer):
            self.preprocessor = Pipeline(steps=[('preprocessor', preprocessor)])
        else:
            self.preprocessor = None
        if str(type(estimator)).split('.')[0] == "<class 'tensorflow":
            self.estimator = estimator
            self.model_type = 'tensorflow'
        else:
            self.estimator = Pipeline(steps=[('estimator', estimator)]) if estimator is not None else None
            self.model_type = 'sklearn'
        # Class atributes:
        self.initial_types_prep = None
        self.initial_types_estimator = None
        self.prep_filename = None
        self.estimator_filename = None
        self.keras_folderpath = None
        self.columns_to_drop = None
        # Class outputs:
        self.output_label = None
        self.output_probability = None
        self.enconded_x = None

    def convert_initial_types(self, x_train: pd.DataFrame, y_train=None, columns_to_drop: list = None, show_types=False):
        self.columns_to_drop = columns_to_drop if columns_to_drop is not None else []
        self.initial_types_prep = convert_initial_types_schema(x_train, self.columns_to_drop)
        x_train_transf = self.preprocessor.fit_transform(x_train, y_train)
        self.initial_types_estimator = [('x_values', FloatTensorType(shape=[None, x_train_transf.shape[1]]))]
        if show_types:
            self.show_initial_types()

    def show_initial_types(self):
        print(f"\nInitial types preprocessing ({len(self.initial_types_prep)}):")
        for init_type in self.initial_types_prep:
            print(init_type[0], str(init_type[1]))
        print(f"\nInitial types estimator ({len(self.initial_types_estimator)}):")
        for init_type in self.initial_types_estimator:
            print(init_type[0], str(init_type[1]))

    def dump_models(self, prep_filename:str = None,
                    estimator_filename:str = None, keras_folderpath:str = None):

        if prep_filename is not None:
            # Convert the preprocessor into ONNX format
            self.prep_filename = f"{prep_filename.replace('.onnx', '')}.onnx"
            if not os.path.exists(os.path.dirname(self.prep_filename)):
                os.makedirs(os.path.dirname(self.prep_filename))
            onx1 = convert_sklearn(self.preprocessor, initial_types=self.initial_types_prep)
            with open(self.prep_filename, "wb") as f_object:
                f_object.write(onx1.SerializeToString())
            print(self.prep_filename, 'saved successfully.')

        if estimator_filename is not None:
            if self.model_type == 'tensorflow':
                # Convert model into keras and TFLite formats:
                if keras_folderpath is not None:
                    self.keras_folderpath = keras_folderpath
                    if not os.path.exists(keras_folderpath):
                        os.makedirs(keras_folderpath)
                    self.estimator.save(keras_folderpath)
                self.estimator_filename = f"{estimator_filename.replace('.tflite', '')}.tflite"
                if not os.path.exists(os.path.dirname(self.estimator_filename)):
                    os.makedirs(os.path.dirname(self.estimator_filename))
                converter = tf.lite.TFLiteConverter.from_keras_model(self.estimator)
                tflite_model = converter.convert()
                with tf.io.gfile.GFile(self.estimator_filename, 'wb') as f:
                    f.write(tflite_model)
            else:
                # Convert the estimator into ONNX format
                self.estimator_filename = f"{estimator_filename.replace('.onnx', '')}.onnx"
                if not os.path.exists(os.path.dirname(self.estimator_filename)):
                    os.makedirs(os.path.dirname(self.estimator_filename))
                onx2 = convert_sklearn(self.estimator, initial_types=self.initial_types_estimator)
                with open(self.estimator_filename, "wb") as f_object:
                    f_object.write(onx2.SerializeToString())
            print(self.estimator_filename, 'saved successfully.')


class ONNXPredictorCLF():
    def __init__(self, prep_filename, estimator_filename):
        # Class inputs:
        if estimator_filename.split('.')[-1] == 'tflite':
            self.model_type = 'tensorflow'
        else:
            self.model_type = 'sklearn'
        self.prep_filename = prep_filename
        self.estimator_filename = estimator_filename
        # Class atributes:
        sess_1 = rt.InferenceSession(prep_filename)
        self.prep_input_types = {x.name: x.type.replace('tensor(', '').replace(')', '') for x in sess_1.get_inputs()}
        # Class outputs:
        self.output_label = None
        self.output_probability = None
        self.enconded_x = None

    def run_predictions(self, x_test: pd.DataFrame):
        # Converting the inputs to the preprocessor and executing the preprocessing stage:
        columns_to_drop = list(x_test.columns[~np.isin(x_test.columns, list(self.prep_input_types.keys()))])
        x_test = x_test.drop(columns=columns_to_drop)
        x_test = convert_input_types(x_test, self.prep_input_types)
        inputs_prep = convert_input_schema(x_test)
        sess_1 = rt.InferenceSession(self.prep_filename)
        onx_prep = sess_1.run(None, inputs_prep)
        self.enconded_x = onx_prep[0]

        if self.model_type == 'tensorflow':
            # Load the TFLite model and allocate tensors.
            interpreter = tf.lite.Interpreter(model_path=self.estimator_filename)
            interpreter.allocate_tensors()
            # Get input and output tensors.
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            outputs_probs = None
            for input_array in onx_prep[0]:
                interpreter.set_tensor(input_details[0]['index'], [input_array])
                interpreter.invoke()
                # The function `get_tensor()` returns a copy of the tensor data.
                # Use `tensor()` in order to get a pointer to the tensor.
                output_data = interpreter.get_tensor(output_details[0]['index'])
                if outputs_probs is None:
                    outputs_probs = output_data
                else:
                    outputs_probs = np.concatenate((outputs_probs, output_data))
            self.output_probability = outputs_probs
            self.output_label = np.argmax(outputs_probs, axis=1)
        else:
            # Compute the prediction with ONNX Runtime
            sess_2 = rt.InferenceSession(self.estimator_filename)
            onx_pred = sess_2.run(None, {'x_values': onx_prep[0]})
            self.output_label = onx_pred[0]
            try:
                self.output_probability = onx_pred[1]
            except IndexError:
                self.output_probability = None
        return self.output_label

class ONNXPredictorREGR():
    def __init__(self, prep_filename, estimator_filename):
        # Class inputs:
        if estimator_filename.split('.')[-1] == 'tflite':
            self.model_type = 'tensorflow'
        else:
            self.model_type = 'sklearn'
        self.prep_filename = prep_filename
        self.estimator_filename = estimator_filename
        # Class atributes:
        sess_1 = rt.InferenceSession(prep_filename)
        self.prep_input_types = {x.name: x.type.replace('tensor(', '').replace(')', '') for x in sess_1.get_inputs()}
        # Class outputs:
        self.output_value = None
        self.enconded_x = None

    def run_predictions(self, x_test: pd.DataFrame):
        # Converting the inputs to the preprocessor and executing the preprocessing stage:
        columns_to_drop = list(x_test.columns[~np.isin(x_test.columns, list(self.prep_input_types.keys()))])
        x_test = x_test.drop(columns=columns_to_drop)
        x_test = convert_input_types(x_test, self.prep_input_types)
        inputs_prep = convert_input_schema(x_test)
        sess_1 = rt.InferenceSession(self.prep_filename)
        onx_prep = sess_1.run(None, inputs_prep)
        self.enconded_x = onx_prep[0]

        if self.model_type == 'tensorflow':
            # Load the TFLite model and allocate tensors.
            interpreter = tf.lite.Interpreter(model_path=self.estimator_filename)
            interpreter.allocate_tensors()
            # Get input and output tensors.
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            outputs_probs = None
            for input_array in onx_prep[0]:
                interpreter.set_tensor(input_details[0]['index'], [input_array])
                interpreter.invoke()
                # The function `get_tensor()` returns a copy of the tensor data.
                # Use `tensor()` in order to get a pointer to the tensor.
                output_data = interpreter.get_tensor(output_details[0]['index'])
                if outputs_probs is None:
                    outputs_probs = output_data
                else:
                    outputs_probs = np.concatenate((outputs_probs, output_data))
            self.output_value = np.argmax(outputs_probs, axis=1)
        else:
            # Compute the prediction with ONNX Runtime
            sess_2 = rt.InferenceSession(self.estimator_filename)
            onx_pred = sess_2.run(None, {'x_values': onx_prep[0]})
            self.output_value = onx_pred[0].reshape(1, -1)[0]
        return self.output_value
