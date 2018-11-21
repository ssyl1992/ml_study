from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
from tensorflow.python.ops import nn

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format





#加载数据
california_housing_dataframe = pd.read_csv("https://download.mlcc.google.cn/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))

#预处理特征
def preprocess_features(california_housing_dateframe):
    selected_features = california_housing_dataframe[
        ["latitude",
         "longitude",
         "housing_median_age",
         "total_rooms",
         "total_bedrooms",
         "population",
         "households",
         "median_income"]]

    # print(selected_features)
    processed_features = selected_features.copy()

    #新增人均room
    processed_features['rooms_per_person'] = (california_housing_dataframe['total_rooms']/california_housing_dataframe['population'])
    return  processed_features


#目标输出预处理
def preprocess_targets(california_housing_dataframe):
    output_targets = pd.DataFrame()

    output_targets['median_house_value'] = (california_housing_dataframe['median_house_value']/1000.0)

    return  output_targets

#训练集
training_examples = preprocess_features(california_housing_dataframe).head(12000) #训练集
training_targets = preprocess_targets(california_housing_dataframe).head(12000)

#验证集
validation_examples = preprocess_features(california_housing_dataframe).tail(5000)
validation_targets = preprocess_targets(california_housing_dataframe).tail(5000)


# Double-check that we've done the right thing.
print("Training examples summary:")
display.display(training_examples.describe())
print("Validation examples summary:")
display.display(validation_examples.describe())

print("Training targets summary:")
display.display(training_targets.describe())
print("Validation targets summary:")
display.display(validation_targets.describe())

#构建特征列
def construct_feature_columns(input_features):

    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])



#输入函数
def my_input_fn(features,targets,batch_size=1,shuffle=True,num_epochs=None):
    # print('call my_input_fn')
    #features = {"latitued":[],"median_income":[]}
    features = {key:np.array(value) for key,value in dict(features).items()}

    #创建数据集
    ds = Dataset.from_tensor_slices((features,targets))

    #将数据集按照batch_size
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds = ds.shuffle(10000)
    #迭代数据集
    features,lables = ds.make_one_shot_iterator().get_next()

    return features,lables

def train_nn_regression_model(learning_rate,
                             steps,
                             batch_size,
                             hidden_units,
                             training_examples,
                             training_targets,
                             validation_examples,
                             validation_targets):

    print('call train_nn_regression_mode')
    periods = 10

    steps_per_period = steps/periods  #输出10次训练结果，每一次输出的迭代的步数

    # 梯度下降
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    #这里的clip_by_norm是指对梯度进行裁剪，通过控制梯度的最大范式，防止梯度爆炸的问题，是一种比较常用的梯度规约的方式。

    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer,5.0)

    #深度神经网络DNN  deap netural nets
    #activation_fn=nn.relu, 默认激活函数是relu
    dnn_regressor = tf.estimator.DNNRegressor(
        feature_columns=construct_feature_columns(training_examples),
        hidden_units=hidden_units,
        optimizer=my_optimizer,
        # activation_fn=nn.sigmoid
    )

    # W = tf.Variable(tf.random_uniform([1], -20.0, 20.0), dtype=tf.float32, name='w')

    #输入函数  training_input_fn： （features，lables）
    training_input_fn = lambda :my_input_fn(training_examples,training_targets['median_house_value'],batch_size=batch_size)

    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                    training_targets["median_house_value"],
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                      validation_targets["median_house_value"],
                                                      num_epochs=1,
                                                      shuffle=False)

    print('training model')
    print("RMSE (on training data):")

    training_rmse = []
    validation_rmse = []

    for periods in range(0,periods):
        dnn_regressor.train(input_fn=training_input_fn, steps=steps_per_period)
        training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

    # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))

    # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (periods, training_root_mean_squared_error))
        print("  period %02d : validation  %0.2f" % (periods, validation_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)

    print("Model training finished.")

    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.show()

    print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
    print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)

    return dnn_regressor,training_rmse,validation_rmse

def linear_scale(series):
    min_val = series.min()
    max_val = series.max()
    scale = (max_val-min_val) / 2.0
    return series.apply(lambda x:((x-min_val)/scale) - 1.0)


def normalize_linear_scale(examples_dataframe):
  """Returns a version of the input `DataFrame` that has all its features normalized linearly."""
  processed_features = pd.DataFrame()
  processed_features["latitude"] = linear_scale(examples_dataframe["latitude"])
  processed_features["longitude"] = linear_scale(examples_dataframe["longitude"])
  processed_features["housing_median_age"] = linear_scale(examples_dataframe["housing_median_age"])
  processed_features["total_rooms"] = linear_scale(examples_dataframe["total_rooms"])
  processed_features["total_bedrooms"] = linear_scale(examples_dataframe["total_bedrooms"])
  processed_features["population"] = linear_scale(examples_dataframe["population"])
  processed_features["households"] = linear_scale(examples_dataframe["households"])
  processed_features["median_income"] = linear_scale(examples_dataframe["median_income"])
  processed_features["rooms_per_person"] = linear_scale(examples_dataframe["rooms_per_person"])
  return processed_features

normalized_dataframe = normalize_linear_scale(preprocess_features(california_housing_dataframe))
normalized_training_examples = normalized_dataframe.head(12000)
normalized_validation_examples = normalized_dataframe.tail(5000)


_ = train_nn_regression_model(
    learning_rate=0.005,
    steps=2000,
    batch_size=70,
    hidden_units=[10, 10],
    training_examples=normalized_training_examples,
    training_targets=training_targets,
    validation_examples=normalized_validation_examples,
    validation_targets=validation_targets)

