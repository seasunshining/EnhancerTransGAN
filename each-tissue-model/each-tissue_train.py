import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Reshape, Lambda, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import keras_tuner as kt
from sklearn.model_selection import train_test_split

base_path = '/work/38liCQ/each-tissue/'

tissue_folders = [folder for folder in os.listdir(base_path) if folder.endswith('_hg38_TE')]

def filter_sequences(sequences):
    valid_chars = set('ATCG')
    return [seq for seq in sequences if all(char in valid_chars for char in seq)]

def encode_sequences(sequences, char_to_int, eos_value):
    encoded = []
    for seq in sequences:
        encoded_seq = [char_to_int[char] for char in seq]
        encoded_seq.append(eos_value)
        encoded.append(np.array(encoded_seq, dtype=int))
    return encoded

def pad_sequences(sequences, max_length, padding_value):
    return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post', value=padding_value)

def build_model(hp, seq_length, num_classes):
    latent_dim = hp.Int('latent_dim', min_value=64, max_value=256, step=64)
    embed_dim = hp.Int('embed_dim', min_value=32, max_value=128, step=32)
    num_heads = hp.Int('num_heads', min_value=4, max_value=16, step=4)

    encoder_input = Input(shape=(seq_length,))
    embedded_input = Embedding(input_dim=num_classes, output_dim=embed_dim, input_length=seq_length)(encoder_input)

    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(embedded_input, embedded_input)
    encoder_output = Flatten()(attention_output)
    z_mean = Dense(latent_dim)(encoder_output)
    z_log_var = Dense(latent_dim)(encoder_output)
    
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling)([z_mean, z_log_var])

    decoder_hidden = Dense(seq_length * embed_dim)(z)
    decoder_hidden = Reshape((seq_length, embed_dim))(decoder_hidden)
    attention_decoded = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(decoder_hidden, decoder_hidden)
    decoder_output = Dense(num_classes, activation="softmax")(attention_decoded)

    vae = Model(inputs=encoder_input, outputs=decoder_output)
    vae.compile(optimizer=Adam(),
                loss=SparseCategoricalCrossentropy(from_logits=False))
    return vae

for tissue_folder in tissue_folders:
    tissue_name = tissue_folder.split('_hg38_TE')[0]
    data_path = os.path.join(base_path, tissue_folder, f"{tissue_name}_train.csv")
    data = pd.read_csv(data_path)
    sequences = data['Sequence'].values
    sequences = filter_sequences(sequences)
    eos_token = 'EOS'
    eos_value = 4
    seq_length = max(len(seq) for seq in sequences) + 1
    num_classes = 5  # A, C, G, T, EOS

    char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3, eos_token: eos_value}
    int_to_char = {0: 'A', 1: 'C', 2: 'G', 3: 'T', eos_value: eos_token}

    X = encode_sequences(sequences, char_to_int, eos_value)
    X_padded = pad_sequences(X, seq_length, eos_value)

    X_train, X_test = train_test_split(X_padded, test_size=0.2, random_state=42)

    tuner = kt.Hyperband(
        lambda hp: build_model(hp, seq_length, num_classes),
        objective='val_loss',
        max_epochs=20,
        factor=3,
        directory='my_dir',
        project_name=f'{tissue_name}_vae-length-enhancer_vae_transformer'
    )

    tuner.search(X_train, X_train, epochs=50, validation_split=0.2)

    best_model = tuner.get_best_models(num_models=1)[0]
    model_save_path = f'{tissue_name}-vae-length-best_enhancer_vae_transformer_model.h5'
    best_model.save(model_save_path)
    print(f"{tissue_name} model has been saved to: {model_save_path}")
