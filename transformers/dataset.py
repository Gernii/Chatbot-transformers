import zipfile
import os
import re
import tensorflow as tf
import tensorflow_datasets as tfds
import json
import wget 
import os.path
from os import path
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

def preprocess_sentence(text, contraction_dict):
  text = text.lower()
  text= re.sub(r"([?.!,])", r" \1 ", text)
  text = re.sub(r'[" "]+', " ", text)
  text = re.sub(r"[^a-zA-Z?.!,']+", " ", text)
  
  contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
  def replace(match):
      
    return contraction_dict[match.group(0)]
  return contraction_re.sub(replace, text)





def tokenize_and_filter(hparams, tokenizer, questions, answers):
  tokenized_questions, tokenized_answers = [], []

  for (question, answer) in zip(questions, answers):
    # tokenize sentence
    sentence1 = hparams.start_token + tokenizer.encode(
        question) + hparams.end_token
    sentence2 = hparams.start_token + tokenizer.encode(
        answer) + hparams.end_token

    # check tokenize sentence length
    if len(sentence1) <= hparams.max_length and len(
        sentence2) <= hparams.max_length:
      tokenized_questions.append(sentence1)
      tokenized_answers.append(sentence2)

  # pad tokenized sentences
  tokenized_questions = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_questions, maxlen=hparams.max_length, padding='post')
  tokenized_answers = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_answers, maxlen=hparams.max_length, padding='post')

  return tokenized_questions, tokenized_answers


def get_dataset(hparams):
  if str(path.exists('./transformers/tennis-corpus')) == 'False':
    print('- Dữ liệu về tennis chưa có sẵn. Bắt đầu tải.')
    url = 'http://zissou.infosci.cornell.edu/convokit/datasets/tennis-corpus/tennis-corpus.zip'
    filedata = wget.download(url)
    with zipfile.ZipFile(filedata,"r") as zip_ref:
      zip_ref.extractall("./transformers")
  with open('./transformers/tennis-corpus/utterances.jsonl', 'r') as json_file:
      json_list = list(json_file)
  print('- Xử lý dữ liệu đầu vào...')
  questions = []
  answers = []
  for json_str in json_list:
      result = json.loads(json_str)
    #   print(f"result: {result}")
    #   print(isinstance(result, dict))
      textQA = result['text']
      textQ = str(result['meta']['is_question'])
      textA = str(result['meta']['is_answer'])
      if textQ == 'True':
        questions.append(preprocess_sentence(textQA, contraction_dict))
        continue
      answers.append(preprocess_sentence(textQA, contraction_dict))
  print('-- Sample question: {}'.format(questions[1]))
  print('-- Sample answer: {}'.format(answers[1]))
#   path_to_zip = tf.keras.utils.get_file(
#       'cornell_movie_dialogs.zip',
#       origin=
#       'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip',
#       extract=True)

#   path_to_dataset = os.path.join(
#       os.path.dirname(path_to_zip), "cornell movie-dialogs corpus")

#   # get movie_lines.txt and movive_conversations.txt
#   lines_filename = os.path.join(path_to_dataset, 'movie_lines.txt')
#   conversations_filename = os.path.join(path_to_dataset,
#                                         'movie_conversations.txt')

#   questions, answers = load_conversations(hparams, lines_filename,
#                                           conversations_filename)
  print('- Đang tạo từ điển cho dataset.')
  tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
      questions + answers, target_vocab_size=2**25)

  hparams.start_token = [tokenizer.vocab_size]
  hparams.end_token = [tokenizer.vocab_size + 1]
  hparams.vocab_size = tokenizer.vocab_size + 2

  questions, answers = tokenize_and_filter(hparams, tokenizer, questions,answers)

  dataset = tf.data.Dataset.from_tensor_slices(({
    'inputs': questions,
    'dec_inputs': answers[:, :-1]
  }, answers[:, 1:]))
  dataset = dataset.cache()
  dataset = dataset.shuffle(len(questions))
  dataset = dataset.batch(hparams.batch_size)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  return dataset, tokenizer
