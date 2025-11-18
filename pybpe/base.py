"""
包含基础的Tokenizer类 和 一些常用的辅助函数。
"""
import unicodedata
    
def get_stats(ids,counts=None):
    """
    给定一个整数列表，返回相邻对的计数字典。
    e.g:[1,2,3,1,2] -> [(1,2):2,(2,3):1,(3,1):1]
    """
    counts = {} if counts is None else counts
    for pair in zip(ids,ids[1:]):
        counts[pair] = counts.get(pair,0) + 1
    return counts

def merge(ids,pair,idx):
    """
    给定整数列表，根据pair进行替换.
    e.g:ids = [1,2,3,1,2],pair=(1,2),idx = 4 -> [4,3,4]
    """
    newids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids)-1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def replace_control_characters(s:str)->str:
    """
    可视化相关
    将字符串中的“控制字符”（不可见或具有特殊格式功能的字符）替换为可打印的 Unicode 转义序列
    e.g:如果你直接 print() 一个包含换行符的 token，它真的会换行.
    """
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C": # 将ch转化为Unicode类别，如果以'C'开头就是控制字符
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}") # 控制字符转化为'\uxxxx'的形式
    return "".join(chars)

def render_token(t:bytes)->str:
    """
    可视化相关
    将字节序列打印成字符串，处理坏数据和控制字符
    """
    s = t.decode('utf-8',errors='replace') # 解码。errors='replace'表示：无法解码时用''代替
    s = replace_control_characters(s)      # 处理控制字符
    return s

class Tokenizer:
    def __init__(self):
        # 默认词汇表大小为256
        self.merges = {}  # (int,int) -> int
        self.pattern = "" # str 正则表达式
        self.special_tokens = {} # str -> int  e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes
    
    def train(self,text,vocab_size,verbose=False):
        raise NotImplementedError
    
    def encode(self,text):
        raise NotImplementedError
    
    def decode(self,ids):
        raise NotImplementedError
    
    def _build_vocab(self):
        vocab = {idx:bytes([idx]) for idx in range(256)}
        for (p0,p1),idx in self.merges.items():
            vocab[idx] = vocab[p0]+vocab[p1]
        for special,idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab
    
    def save(self,file_prefix):
        """
        存储训练好的分词器模型
        生成：.model 给机器读取；使用load()方法读取
             .vocab 给人类查看。
        """
        model_file = file_prefix + '.model'
        with open(model_file,'w') as f:
            f.write('pybpe v1\n')  # 版本标识
            f.write(f'{self.pattern}\n') # 正则表达式
            f.write(f'{len(self.special_tokens)}\n')  # 特殊token数量，方便遍历
            for special,idx in self.special_tokens.items(): # 写入所有特殊token
                f.write(f'{special} {idx}\n')
            for idx1,idx2 in self.merges: # 写入合并规则
                f.write(f'{idx1} {idx2}\n')
        
        vocab_file = file_prefix + '.vocab'
        inverted_merges = {idx:pair for pair,idx in self.merges.items()} # 反转
        with open(vocab_file,'w',encoding='utf-8') as f:
            for idx,token in self.vocab.items(): # 遍历词表
                s = render_token(token) # token转字符

                if idx in inverted_merges: # 如果这个token是合并得到的 写入合并结构
                    idx0,idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f'[{s0}][{s1}] -> [{s}] {idx}\n')
                else: # 基础token直接写入
                    f.write(f'[{s}] {idx}\n')
    
    def load(self,model_file):
        """
        加载使用save()方法导出的.model文件
        """
        assert model_file.endswith('.model')
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file,'r',encoding='utf-8') as f:
            version = f.readline().strip()
            assert version == 'pybpe v1'
            self.pattern = f.readline().strip()
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special,special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            for line in f:
                idx1,idx2 = map(int,line.split())
                merges[(idx1,idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()    
