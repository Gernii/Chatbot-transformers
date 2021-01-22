
import tensorflow as tf
import os.path
from os import path
tf.random.set_seed(1234)
from transformers.model import transformer
from transformers.dataset import get_dataset, preprocess_sentence


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  
  def __init__(self, hparams, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    self.d_model = tf.cast(hparams.d_model, dtype=tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * self.warmup_steps**-1.5
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
class Args():
  max_length = 20
  batch_size = 256
  num_layers = 2
  num_units = 512
  d_model = 256
  num_heads = 8
  dropout = 0.1
  activation = 'relu'
  epochs = 20
  evaluate = 'yes'
  train = 'no'
hparams = Args()

contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", 
                    "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", 
                    "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", 
                    "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", 
                    "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", 
                    "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                    "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not",
                    "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", 
                    "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", 
                    "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", 
                    "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                    "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", 
                    "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", 
                    "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have",
                    "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
                    "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  
                    "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", 
                    "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", 
                    "why've": "why have", "will've": "will have", "won't": "will not",  "would've": "would have", "wouldn't": "would not", 
                    "'ve": "have", "y'all": "you all ", "'d": "would","d've": " would have","'re": "are",
                    "you'd": "you would",  "you'll": "you will",  "you're": "you are", 
                    "you've": "you have"}

def main():
  print('Lấy dữ liệu.')
  dataset, tokenizer = get_dataset(hparams)
  print('Hoàn tất lấy dữ liệu.')
  print('Chuẩn bị mô hình...')
  model = transformer(hparams)

  optimizer = tf.keras.optimizers.Adam(
      CustomSchedule(hparams), beta_1=0.9, beta_2=0.98, epsilon=1e-9)

  cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')
  def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
      return optimizer._decayed_lr(tf.float32) 
    return lr

  def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, hparams.max_length - 1))
    loss = cross_entropy(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32)
    loss = tf.multiply(loss, mask)
    return tf.reduce_mean(loss)

  def accuracy(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, hparams.max_length - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
  lr_metric = get_lr_metric(optimizer)
  model.compile(optimizer, loss=loss_function, metrics=['accuracy', lr_metric])
  print('Hoàn tất chuẩn bị mô hình.')
  filename = 'weights.h5'
  if str(path.exists('./weights.h5')) == 'True' :
    print('Đang tải pre-trained model.')
    model.load_weights(filename)
    print('Tải pre-trained model hoàn tất.')
    return model, dataset, tokenizer
  elif str(path.exists('./weights.h5')) == 'False':
    print('Khong co pretrained-model.')
    'Tải check point để train.'
    checkpoint_filepath = './pretrained/'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath,
      save_weights_only=True,
      monitor='accuracy',
      mode='max',
      save_best_only=True,
      save_freq = 'epoch')

    if  str(path.exists('./pretrained/checkpoint')) == 'True':
      model.load_weights(checkpoint_filepath)
      model.fit(dataset, epochs=hparams.epochs, callbacks= [model_checkpoint_callback])
      model.load_weights(checkpoint_filepath)
      filename = 'weights.h5'
      model.save_weights(filename)
    else:
      print('Thiếu file check point.')
      model.fit(dataset, epochs=hparams.epochs, callbacks= [model_checkpoint_callback])
    return model, dataset, tokenizer
model, dataset, tokenizer = main()


def inference(sentence, model, tokenizer):
  sentence = preprocess_sentence(sentence, contraction_dict)

  sentence = tf.expand_dims(
      hparams.start_token + tokenizer.encode(sentence) + hparams.end_token,
      axis=0)

  output = tf.expand_dims(hparams.start_token, 0)

  for i in range(hparams.max_length):
    predictions = model(inputs=[sentence, output], training=False)

    # select the last word from the seq_len dimension
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # return the result if the predicted_id is equal to the end token
    if tf.equal(predicted_id, hparams.end_token[0]):
      break

    # concatenated the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)


def predict(sentence):
  prediction = inference(sentence, model = model, tokenizer  =  tokenizer )

  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

  return predicted_sentence



def train_model():
  'Tải check point để train.'
  checkpoint_filepath = './pretrained/'
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='accuracy',
    mode='max',
    save_best_only=True,
    save_freq = 'epoch')

  if  str(path.exists('./pretrained/checkpoint')) == 'True':
    model.load_weights(checkpoint_filepath)
    model.fit(dataset, epochs=hparams.epochs, callbacks= [model_checkpoint_callback])
    model.load_weights(checkpoint_filepath)
    filename = 'weights.h5'
    model.save_weights(filename)
  else:
      print('Thiếu file check point.')
      model.fit(dataset, epochs=hparams.epochs, callbacks= [model_checkpoint_callback])
def Get_Hyper():
      print('Kích thước 1 câu - max_length:',hparams.max_length)
      print('Batch size - batch_size:',hparams.batch_size)
      print('Số lớp "encoder layer" và "decoder layer" - num_layers:',hparams.num_layers)
      print('số lượng Units - num_units:',hparams.num_units)
      print('Số chiều Model - d_model:', hparams.d_model)
      print('Số lượng head trong Multi-head attention - num_heads:',hparams.num_heads)
      print('Dropout - dropout:',hparams.dropout)
      print('Activation - activation:', hparams.activation)
      print('Epochs - epochs:', hparams.epochs)
      print('Nếu muốn đổi thông tin Hyperparameter, vui lòng mở file main.py và đổi thông tin trong class Args rồi restart runtime')




