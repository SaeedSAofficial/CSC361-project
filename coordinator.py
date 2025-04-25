import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from functools import lru_cache
import tensorflow as tf
import numpy as np

class CoordinatorAgent:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self._embedder = None  # Lazy loading
        self.model = self.build_model()
        self.trained = False
        
    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder
    
    def build_model(self):
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(384,), dtype=tf.float32),

            # First Dense Block
            tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.45),

            # Second Dense Block
            tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.45),

            # Output Layer
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    
    @lru_cache(maxsize=128)
    def _encode_text(self, text):
        return self.embedder.encode(text)
    
    def preprocess(self, X_raw):
        if isinstance(X_raw, str):
            return np.array([self._encode_text(X_raw)], dtype=np.float32)
        embeddings = [self._encode_text(text) for text in X_raw]
        return np.array(embeddings, dtype=np.float32)
    
    def train(self, data, epochs=10, batch_size=32):
        queries, labels = zip(*data)
        y = np.array([1 if label == "summarizer" else 0 for label in labels], dtype=np.float32)
        X = self.preprocess(queries)

        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Try to use all available GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                # Enable mixed precision for compatible GPUs
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
            except RuntimeError:
                pass
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True
        )
        
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=0  # Changed from 1 to 0 to remove output
        )
        self.trained = True
    
    def predict_agent(self, query):
        X = self.preprocess(query)
        pred = self.model.predict(X, verbose=0)[0][0]
        return "summarizer" if pred > 0.65 else "tutor"
    
    def predict_batch(self, queries):
        X = self.preprocess(queries)
        preds = self.model.predict(X, verbose=0)
        return ["summarizer" if p > 0.65 else "tutor" for p in preds]
    
    def save(self, path):
        self.model.save(path, save_format='tf')  # Save full model including optimizer & structure
        
    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        self.trained = True