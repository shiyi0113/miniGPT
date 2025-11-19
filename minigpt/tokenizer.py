"""
RustBPE Tokenizer 的 Python 包装器。
它负责调用底层的 rustbpe 扩展进行训练，并使用 tiktoken 进行推理
"""

import os
import pickle
import tiktoken
from functools import lru_cache  # 函数修饰器 最近最少使用缓存 第二次调用该函数的时候直接用缓存的内容

import rustbpe

SPECIAL_TOKENS = [
    "<|bos|>",             # 文档开始 (Beginning of Sequence)
    "<|user_start|>",      # 用户发言开始
    "<|user_end|>",
    "<|assistant_start|>", # 助手发言开始
    "<|assistant_end|>",
    "<|python_start|>",    # 助手调用 Python 工具
    "<|python_end|>",
    "<|output_start|>",    # 工具返回结果
    "<|output_end|>",
]

# 分词正则模式
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RustBPETokenizer:
    def __init__(self,enc,bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)
    
    @lru_cache(maxsize=32)
    def encode_special(self,text):
        return self.enc.encode_single_token(text)
    
    @classmethod
    def train_from_iterator(cls,text_iterator,vocab_size):

        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >=256,f'Vocab size too small:{vocab_size_no_special}'

        tokenizer.train_from_iterator(text_iterator,vocab_size_no_special,pattern=SPLIT_PATTERN)
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k):v for k,v in mergeable_ranks_list}

        tokens_offset = len(mergeable_ranks)
        special_tokens = {name:tokens_offset + i for i,name in enumerate(SPECIAL_TOKENS)}

        enc = tiktoken.Encoding(
            name = 'rustbpe',
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return cls(enc,"<|bos|>")
    
    @classmethod
    def from_directory(cls,tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir,'tokenizer.pkl')
        with open(pickle_path,'rb') as f:
            enc = pickle.load(f)
        return cls(enc,'<|bos|>')
    # 词汇表大小
    def get_vocab_size(self):
        return self.enc.n_vocab
    # 开始词位置
    def get_bos_token_id(self):
        return self.bos_token_id
    # 特殊词表
    def get_special_tokens(self):
        return self.enc.special_tokens_set
    # 编码
    def encode(self,text,prepend=None,append=None,num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend,int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids
    # 解码token序列
    def decode(self, ids):
        return self.enc.decode(ids)
    # 这个token是哪个词
    def id_to_token(self,id):
        return self.enc.decode([id])
    # 存分词器
    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

# 便捷函数：获取~/.cache/minigpt/tokenizer文件夹内的分词器
def get_tokenizer():
    from minigpt.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    return RustBPETokenizer.from_directory(tokenizer_dir)  
