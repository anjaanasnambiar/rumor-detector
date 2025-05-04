import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

df = pd.read_csv('rumour_dataset.csv').dropna(subset=['text', 'is_rumor'])
df['is_rumor'] = df['is_rumor'].astype(int)
df = df.sample(n=1000, random_state=42)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].values, df['is_rumor'].values,
    test_size=0.2, stratify=df['is_rumor'], random_state=42
)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels,
    test_size=0.2, stratify=train_labels, random_state=42
)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_examples(texts, labels, tokenizer, max_length=MAX_LENGTH):
    input_ids, attention_masks = [], []
    for text in texts:
        tokenized = tokenizer.encode_plus(
            text, max_length=max_length, padding='max_length',
            truncation=True, return_attention_mask=True
        )
        input_ids.append(tokenized['input_ids'])
        attention_masks.append(tokenized['attention_mask'])
    return {
        'input_ids': np.array(input_ids),
        'attention_mask': np.array(attention_masks)
    }, np.array(labels)

train_dataset = encode_examples(train_texts, train_labels, tokenizer)
val_dataset = encode_examples(val_texts, val_labels, tokenizer)
test_dataset = encode_examples(test_texts, test_labels, tokenizer)

train_tf_dataset = tf.data.Dataset.from_tensor_slices(train_dataset).shuffle(1000).batch(BATCH_SIZE)
val_tf_dataset = tf.data.Dataset.from_tensor_slices(val_dataset).batch(BATCH_SIZE)
test_tf_dataset = tf.data.Dataset.from_tensor_slices(test_dataset).batch(BATCH_SIZE)

class BertLSTMModel(tf.keras.Model):
    def __init__(self, dropout_rate=0.2):
        super(BertLSTMModel, self).__init__()
        self.bert = TFBertModel.from_pretrained('bert-base-uncased')
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dense = tf.keras.layers.Dense(32, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        bert_outputs = self.bert(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )[0]
        x = self.bilstm(bert_outputs)
        x = self.dropout1(x, training=training)
        x = self.dense(x)
        x = self.dropout2(x, training=training)
        return self.output_layer(x)

def predict_rumor(texts, model, tokenizer):
    inputs, _ = encode_examples(texts, [0] * len(texts), tokenizer)
    predictions = model.predict(inputs).flatten()
    results = []
    for i, text in enumerate(texts):
        results.append({
            'text': text,
            'probability': float(predictions[i]),
            'is_rumor': bool(predictions[i] > 0.5)
        })
    return results

def main():
    model = BertLSTMModel()
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
    ]

    model.fit(train_tf_dataset, validation_data=val_tf_dataset, epochs=EPOCHS, callbacks=callbacks)
    test_loss, test_acc = model.evaluate(test_tf_dataset)
    print(f"Test Accuracy: {test_acc:.4f}")

    y_pred_probs = model.predict(test_tf_dataset).flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)
    print(classification_report(test_labels, y_pred, target_names=['Non-Rumor', 'Rumor']))

    sample_texts = [
        "Police have confirmed three people involved in the attack.",
        "Charlie Hebdo became well known for publishing controversial cartoons."
    ]
    results = predict_rumor(sample_texts, model, tokenizer)
    for result in results:
        print(f"\nText: {result['text']}")
        print(f"Prediction: {'Rumor' if result['is_rumor'] else 'Not Rumor'}")
        print(f"Probability: {result['probability']:.4f}")

if __name__ == "__main__":
    main()
