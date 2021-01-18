## there are funcs for data preprocessing.

from sklearn.utils import shuffle
import pandas as pd
import numpy as np

## function to convert sequence strings into k-mer words.
def getKmers(seqce, size=6):
    if len(seqce) <= size:
        print('The sequence is too short.\n')
        return []

    res = [seqce[x:x+size].lower() for x in range(len(seqce) - size + 1)]
    return res

## function to translate words col to seq col.
def seq2words(df, size = 6):
    df['words'] = df.apply(lambda x: getKmers(x['sequence'], size), axis=1)
    res = df.drop('sequence', axis=1)
    return res

## function to create train and label
def df2text(df_input):
    df = pd.DataFrame(df_input)
    df = shuffle(df)
    df['text'] = df['words'].apply(lambda x: ' '.join(x))
    text = list(df['text'])
    y = list(df['class'])
    
    return np.array(text), np.array(y)

## The function aggregate all the datapreprocess func
def dataPreprocess(df, sz):
    df1 = seq2words(df, sz)
    _text, _y = df2text(df1)
    
    return _text, _y