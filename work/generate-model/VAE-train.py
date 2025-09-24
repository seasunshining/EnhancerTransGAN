import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Conv1D, MaxPooling1D, Flatten, Reshape, Lambda, MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import keras_tuner as kt
from sklearn.model_selection import train_test_split

data_path = '/work/38liCQ/each-tissue/Human_Renal_hg38_TE/Human_Renal_train.csv'
save_path = '/work/40model/model-get/Human_Renal'

data = pd.read_csv(data_path)
sequences = data['Sequence'].values

def filter_sequences(sequences):
    valid_chars = set('ATCG')
    return [seq for seq in sequences if all(char in valid_chars for char in seq)]

sequences = filter_sequences(sequences)

seq_length = max(len(seq) for seq in sequences)
num_classes = 4  # A, C, G, T

char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
int_to_char = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

def encode_sequences(sequences, seq_length, char_to_int):
    encoded = np.zeros((len(sequences), seq_length), dtype=int)
    for i, seq in enumerate(sequences):
        for j, char in enumerate(seq):
            encoded[i, j] = char_to_int.get(char, 0)
    return encoded

X = encode_sequences(sequences, seq_length, char_to_int)

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

def to_int_labels(X):
    return X

X_train = to_int_labels(X_train)
X_test = to_int_labels(X_test)

def build_vae_transformer_model(hp):
    inputs = Input(shape=(seq_length,))
    x = Embedding(input_dim=num_classes, output_dim=hp.Int('embedding_dim', min_value=16, max_value=64, step=16))(inputs)
    
    # Ensure sequence length consistency
    x = Conv1D(filters=hp.Int('filters', min_value=16, max_value=64, step=16), kernel_size=hp.Int('kernel_size', min_value=3, max_value=5, step=1), padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Transformer Encoder
    x = MultiHeadAttention(num_heads=hp.Int('num_heads', min_value=2, max_value=8, step=2), key_dim=hp.Int('key_dim', min_value=16, max_value=64, step=16))(x, x)
    x = LayerNormalization()(x)
    
    x = Flatten()(x)
    x = Dense(hp.Int('latent_dim', min_value=32, max_value=128, step=32))(x)
    
    # Latent space
    z_mean = Dense(hp.Int('latent_dim', min_value=32, max_value=128, step=32))(x)
    z_log_var = Dense(hp.Int('latent_dim', min_value=32, max_value=128, step=32))(x)
    
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    z = Lambda(sampling)([z_mean, z_log_var])
    
    # Decoder
    decoder_inputs = Input(shape=(hp.Int('latent_dim', min_value=32, max_value=128, step=32),))
    x = Dense(seq_length * num_classes, activation='relu')(decoder_inputs)
    x = Reshape((seq_length, num_classes))(x)
    x = Conv1D(filters=hp.Int('filters', min_value=16, max_value=64, step=16), kernel_size=hp.Int('kernel_size', min_value=3, max_value=5, step=1), padding='same', activation='softmax')(x)
    
    # Models
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    decoder = Model(decoder_inputs, x, name='decoder')
    
    outputs = decoder(encoder(inputs)[2])
    
    vae = Model(inputs, outputs, name='vae')
    
    reconstruction_loss = SparseCategoricalCrossentropy(from_logits=True)(inputs, outputs)
    kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    
    return vae

def build_model(hp):
    vae = build_vae_transformer_model(hp)
    vae.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')))
    return vae

tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=30,
    hyperband_iterations=2,
    directory='keras_tuner_dir',
    project_name='vae_transformer'
)

tuner.search(X_train, X_train, epochs=30, validation_data=(X_test, X_test), batch_size=32)

# Save the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.save(f'{save_path}/VAE-Transformer-enhancer_sequence_generator_tuning/VAE-Transformer-best_model.h5')

print("The best model has been saved.")

