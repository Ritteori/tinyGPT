import pandas as pd
import os
from collections import Counter
import torch
from tqdm import tqdm

class BPE:
    def __init__(self, data_dir, train_data_path):
        dataset = pd.read_parquet(os.path.join(data_dir, train_data_path))

        word_counter = Counter()
        for row in dataset.values:
            for sentence in row:
                for w in sentence.strip().lower().split():
                    word_counter[w + "</w>"] += 1

        self.word_counter = word_counter
        
        self.chars = sorted({c for w in word_counter for c in w})
        self.tokens = {}
        for w, f in word_counter.most_common(30000):
            self.tokens[w] = f

        self.tokens = {tuple(key):value for key,value in self.tokens.items()}
        self.merges = []
        
    def make_pairs(self,tokens):
        """Make pairs

        Args:
            tokens (tuple):  
            ('o', 'n', 'l', 'y'): 2061,
            ('m', 'o', 's', 't'): 2027,
            ('t', 'h', 'r', 'e', 'e'): 1976,
            
        Out:
            (dict(tuple:freq)):
                ('t', 'h'): 221380,
                ('h', 'e'): 194235,
                ('o', 'f'): 61924,
            best_pair (tuple): ('t', 'h')
        """
        
        pairs_freq = Counter()
        
        for word_tuple, freq in tokens.items():
            
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i+1])
                pairs_freq[pair] += freq
                
        if not pairs_freq:
            return None, None
        
        best_pair = max(pairs_freq,key=pairs_freq.get)
        
        return pairs_freq, best_pair
    
    def concat_best_pair(self,tokens,best_pair,merges):
        """Concat best pair in every word

        Args:
            tokens (dict(tuple:int)): base words dict
            best_pair (tuple): ('t', 'h')
            merges (list[tuples]): [('t', 'h'), ('b', 'e'), ...]
            
        Out:
            (dict(tuple:freq):
                ('th',e): 221380,
                ('h', 'i', 'm'): 194235,
                ('o', 'f'): 61924,
            merges (list[tuples]): [('t', 'h'), ('b', 'e'), ...]
        """
        
        out = {}

        for key,value in tokens.items():
            
            if len(key) <= 1:
                out[key] = value
                continue
            
            new_word = list()
            i = 0
            length = len(key)
            while i < length:
                if i < length - 1 and (key[i], key[i+1]) == best_pair:
                    new_word.append(''.join(best_pair)) 
                    i += 2
                else:
                    new_word.append(key[i])
                    i+=1
                    
            out[tuple(new_word)] = value
        
        merges.append(''.join(best_pair))
        
        return out,merges
    
    def train_bpe(self,num_iterations):
        """_summary_

        Args:
            num_iterations (int): Count of merges and size of vocab size without base symbols
        """
        
        tokens = self.tokens
        merges = self.merges
        
        for _ in tqdm(range(num_iterations),desc='Training bpe...'):
            
            pairs_freq, best_pair = self.make_pairs(tokens)
            updated_tokens, updated_merges= self.concat_best_pair(tokens, best_pair,merges)
            
            tokens = updated_tokens
            merges = updated_merges
            
        return tokens, self.merges
    
    def save(self,path):
        """_summary_

        Args:
            path (_type_): _description_
        """
        
        vocab = ['<unk>'] + list(self.chars) + self.merges
        stoi = {tok: i for i, tok in enumerate(vocab)}
        itos = {i: tok for tok, i in stoi.items()}
        
        state = {
            "merges": self.merges,
            "vocab": vocab,
            "stoi": stoi,
            "itos": itos,
        }
        
        torch.save(state, path)
        
    def encode(self, sentence, stoi):
        """_summary_

        Args:
            sentence (_type_): _description_
            stoi (_type_): _description_
        """
        
        words = sentence.strip().lower().split()
        
        out_tokens = []
        for word in words:
            letters = list(word) + ['</w>']
            
            for merge in self.merges:
                new_letters = []
                i = 0
                
                while i < len(letters):
                    if i < len(letters)-1 and letters[i] + letters[i+1] == merge:
                        new_letters.append(merge)
                        i += 2
                    else:
                        new_letters.append(letters[i])
                        i += 1
                        
                letters = new_letters
            
            out_tokens.append([stoi.get(letter, stoi['<unk>']) for letter in letters])
            
        flat = [item for sublist in out_tokens for item in sublist]
        
        return flat
         
    def decode(self, tokens, itos):
        """_summary_

        Args:
            tokens (_type_): _description_
            itos (_type_): _description_
        """
        
        out_str = []

        for token in tokens:
            
            char = itos[token]
            
            if char.find('</w>') == -1:
                out_str.append(char)
            else:
                if len(char) > len('</w>'):
                    out_str.append(char[:char.find('</w>')])
                    out_str.append(' ')
                else:
                    out_str.append(' ')

        return ''.join(out_str)