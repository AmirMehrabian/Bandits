import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def model_builder(context_dim, num_actions, output_size=1):
    reg_coef = 0.0000
    reg = tf.keras.regularizers.l2(reg_coef)
    input_dim = context_dim + num_actions
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation='relu', kernel_regularizer=reg, bias_regularizer=reg),
        layers.Dense(32, activation='relu', kernel_regularizer=reg, bias_regularizer=reg),
        layers.Dense(output_size)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model


def context_builder(corr_vec, power_jn_db, power_tn_db):
    phi_vec = np.degrees(np.acos(np.abs(corr_vec))) / 90
    return np.array([phi_vec[0], power_jn_db, power_tn_db])


def context_builder_5features(corr_vec, power_jn_db, power_tn_db):
    phi_vec = np.degrees(np.acos(np.abs(corr_vec))) / 90
    return np.array([np.abs(corr_vec)[0], power_jn_db, power_tn_db, phi_vec[0], phi_vec[0] * 90])


def context_builder_10features(corr_vec, power_jn_db, power_tn_db, index_action, config_dict):
    action = config_dict['action_set'][index_action]
    max_reward = config_dict['num_data_symbols'] - ((action - 1) * config_dict['num_pilot_symbols'])
    max_reward_norm = max_reward / config_dict['num_data_symbols']
    phi_vec = np.degrees(np.acos(np.abs(corr_vec))) / 90
    rho = np.abs(corr_vec)[0]
    sir_db = power_tn_db - power_jn_db

    return np.array([1, sir_db, action, action * sir_db,
                     action * rho, rho, power_jn_db, power_tn_db,
                     phi_vec[0], phi_vec[0] * 90, max_reward_norm])


def model_feeder(context, action_index, number_actions):
    one_hot_action = tf.one_hot(action_index, number_actions)
    tf_context = tf.convert_to_tensor(context, dtype=tf.float32)
    tf_input = tf.concat([tf_context, one_hot_action], axis=0)
    return tf_input.numpy().reshape(1, -1)


def model_feeder_no_action(context):
    tf_context = tf.convert_to_tensor(context, dtype=tf.float32)
    return tf_context.numpy().reshape(1, -1)


def epsilon_greedy(epsilon, q_values):
    if np.random.rand() < epsilon:
        return np.random.randint(len(q_values))
    else:
        return np.argmax(q_values)


class ReplayBuffer:
    def __init__(self, capacity, input_model_size, output_model_size=1):
        self.capacity = capacity
        self.replay_input_buffer = np.array([]).reshape(-1, input_model_size)
        self.replay_output_buffer = np.array([]).reshape(-1, output_model_size)

    def add_to_buffer(self, input_sample, output_sample):
        if self.replay_input_buffer.shape[0] < self.capacity:
            self.replay_input_buffer = np.concatenate([self.replay_input_buffer, input_sample], axis=0)
            self.replay_output_buffer = np.concatenate([self.replay_output_buffer, output_sample], axis=0)

        elif self.replay_input_buffer.shape[0] >= self.capacity:
            self.replay_input_buffer = np.concatenate([self.replay_input_buffer[1:], input_sample], axis=0)
            self.replay_output_buffer = np.concatenate([self.replay_output_buffer[1:], output_sample], axis=0)

    def sample_from_buffer(self, batch_size):
        if batch_size > self.replay_input_buffer.shape[0]:
            batch_size = self.replay_input_buffer.shape[0]

        indices = np.random.choice(self.replay_input_buffer.shape[0], size=batch_size, replace=False)
        return self.replay_input_buffer[indices], self.replay_output_buffer[indices]
