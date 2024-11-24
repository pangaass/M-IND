from torch.utils.data import Dataset
import random
import numpy as np
import torch
from dataclasses import dataclass
import pickle
import json
import re
import codecs
import pinyin 
from tool.util import *
from tool.is_chinese import *
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union





LABEL_TOKEN = '<label_token>'
EMBED_TOKEN = '<emb_token>'
GRAPH_TOKEN = '<graph_token>' 
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

#TO BE ADDED
END_OF_TEXT = '<eot>'
END_OF_GRAPH = '<eog>'
END_OF_EMB = '<eoe>'
TRAINABLE_SPECIAL_TOKENS = [END_OF_TEXT,END_OF_GRAPH,END_OF_EMB,LABEL_TOKEN]


special_token_dict = {'additional_special_tokens':TRAINABLE_SPECIAL_TOKENS+[EMBED_TOKEN,GRAPH_TOKEN]}
def decoding(string):
    if re.search(r'\\u[0-9a-fA-F]{4}',string):
        return codecs.decode(string, 'unicode-escape')
    return string

def get_pinyin(name):
    if name == re.search(u'[\u4e00-\u9fff]', name) is not None:
        return pinyin.get(name, delimiter=" ", format="strip")
    return name

def is_not_none(inputs):
    if isinstance(inputs,str):
        return inputs != ''
    elif isinstance(inputs,list):
        return len(inputs) != 0
    else:
        return inputs is not None

def cut_to_target_len(text, target_len):
    if is_not_none(text):
        return " ".join(text.split(" ")[:target_len])
    else:
        return ""

class INDWithAuthorDataset(Dataset):
    def __init__(self,dataset, tokenizer, data_args, model_args , mode = "train",ptm_tokenizer = None, use_graph = False, use_emb = False ):
        super(INDWithAuthorDataset, self).__init__()
        self.use_graph = use_graph
        self.use_emb = use_emb

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.model_args = model_args
        self.author, self.pub = dataset 
        self.ptm_tokenizer = ptm_tokenizer
        self.instruct = "Identify the abnormal text from the text collection according to the following rules:\n Here is a collection of paper titles: \n ### {} \n Determine whether the following is true and answer 'Yes' or 'No'. {}"
        self.local_instruct = "\"{}\" belongs to the main cluster. "+ LABEL_TOKEN + "." 
        self.instruct_length = len(self.tokenizer.tokenize(self.instruct+self.local_instruct))
        if use_graph:
            with open(self.model_args.graph_path, 'rb') as f:
                self.graph_emb = pickle.load(f)
        self.data = []
        self.mode = mode
        if self.mode == "train":
            if data_args.sample :
                for _ in range(4):
                    for key in self.author.keys():
                        if len(self.author[key]['outliers']) < len(self.author[key]['normal_data']): 
                            pos_set = random.sample(self.author[key]['normal_data'],len(self.author[key]['outliers']))
                            neg_set = self.author[key]['outliers']
                        elif len(self.author[key]['outliers']) > len(self.author[key]['normal_data']):
                            neg_set = random.sample(self.author[key]['outliers'],len(self.author[key]['normal_data']))
                            pos_set = self.author[key]['normal_data']
                        else:
                            neg_set = self.author[key]['outliers']
                            pos_set = self.author[key]['normal_data']
                        set = neg_set+pos_set
                        random.shuffle(set)
                        for i in range(len(set)//model_args.packing_size):
                            sampled_pack = set[i*model_args.packing_size:(i+1)*model_args.packing_size]
                            labels = [int(i in self.author[key]['normal_data']) for i in sampled_pack]
                            self.data.append({
                                "author": key,
                                "pubs":sampled_pack,
                                "labels":labels
                            })
            else:
                for key in self.author.keys():
                    set = self.author[key]['outliers']+self.author[key]['normal_data']
                    random.shuffle(set)
                    for i in range(len(set)//model_args.packing_size):
                        sampled_pack = set[i*model_args.packing_size:(i+1)*model_args.packing_size]
                        labels = [int(i in self.author[key]['normal_data']) for i in sampled_pack]
                        self.data.append({
                            "author": key,
                            "pubs":sampled_pack,
                            "labels":labels
                        })
            random.shuffle(self.data)
        else:
            for key in self.author.keys():
                if 'papers' in self.author[key]:
                    set = self.author[key]['papers']
                elif 'normal_data' in self.author[key]:
                    set = self.author[key]['normal_data']+self.author[key]['outliers']
                for i in range(0, len(set), model_args.packing_size):
                    sampled_pack = set[i:i + model_args.packing_size]
                    self.data.append({
                        "author": key,
                        "pubs":sampled_pack,
                    })
        if mode != "train" and data_args.sorted_file is not None:
            with open(data_args.sorted_file,'r') as f:
                self.sorted_file = json.load(f)
        else:
            self.sorted_file = None
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        author_id = self.data[index]['author']
        if self.mode == "train":
            normal_data = self.author[author_id]['normal_data']
            outliers = self.author[author_id]['outliers']
            profile = normal_data + outliers
        else:
            profile = self.author[self.data[index]['author']]['papers']
        if self.mode == 'train':
            random.shuffle(profile)
        # random drop papers
        if self.mode == "train" and self.model_args.papers_drop_ratio > 0:
            profile = random.sample(profile,int(len(profile) * (1 - self.model_args.papers_drop_ratio)))    
        globals = profile[:300] #if self.model_args.input_type == "text" else profile[:1000]
        locals = self.data[index]['pubs']
        
        global_feature_mask = None
        local_feature_mask = None    
        
        if self.mode == "train" and self.model_args.feature_drop_prob is not None:
        # random_drop_mask
            prob = [float(i) for  i in self.model_args.feature_drop_prob.split('_')]
            # global_feature_mask = [generate_random_mask(len(globals),prob[i]) for i in range(len(prob))]
            local_feature_mask = [generate_random_mask(len(locals),prob[i]) for i in range(len(prob))]

            # globals_txt = self._generate_random_mask_context(global_feature_mask, [self.pub[p]['title'] for p in globals])
            globals_txt = [(GRAPH_TOKEN if self.use_graph else '')+ \
                        (EMBED_TOKEN if self.use_emb else '')+ \
                        (self.pub[p]['title']) for p in globals
                        ]
            
            locals_txt = self._generate_random_mask_context(local_feature_mask, [self.pub[p]['title'] for p in locals])
        else:
            globals_txt = [
                            (GRAPH_TOKEN if self.use_graph else '')+ \
                            (EMBED_TOKEN if self.use_emb else '')+ \
                            (self.pub[p]['title']) if self.model_args.use_text else ''\
                            for p in globals
                            ]
                
        locals_txt = [ 
                        (GRAPH_TOKEN if self.use_graph else '')+\
                        (EMBED_TOKEN if self.use_emb else '') + \
                        (self.pub[p]['title']) if self.model_args.use_text else '' \
                        for p in locals
                        ]
        local_instruct = "# ".join([self.local_instruct.format(i) for i in locals_txt])
        locals_length = len(self.tokenizer.tokenize(local_instruct))
        # cut global to target length
        cut_num = self._get_cut_num(globals_txt,self.data_args.max_source_length-1000-locals_length)

        globals = globals[:cut_num]
        globals_txt = globals_txt[:cut_num]
        global_instruct = '#'.join(globals_txt)


        context = self.instruct.format(global_instruct,local_instruct)
        
        llm_inputs = self.tokenizer(context, return_tensors='pt',add_special_tokens=True, truncation=True,max_length=self.data_args.max_source_length)
        
        graph_emb = None
        text_inputs = None
        if self.use_graph:
            if global_feature_mask is not None:
                global_graph_emb = [self.graph_emb[author_id][globals[i]] for i in range(len(globals)) if global_feature_mask[1][i]==1]
            else:
                global_graph_emb = [self.graph_emb[author_id][globals[i]] for i in range(len(globals))]
            if local_feature_mask is not None:
                local_graph_emb = [self.graph_emb[author_id][locals[i]] for i in range(len(locals)) if local_feature_mask[1][i]==1]
            else:
                local_graph_emb = [self.graph_emb[author_id][locals[i]] for i in range(len(locals))]

            graph_emb = global_graph_emb + local_graph_emb
            graph_emb = torch.tensor(graph_emb,dtype=torch.float32)
            graph_emb = torch.cat([graph_emb,torch.tensor(self.graph_emb[author_id]['graph'],dtype=torch.float32).expand(graph_emb.shape[0],-1)],dim=-1) #to debuged
        if self.use_emb:
            if global_feature_mask is not None:
                global_ptm_papers = [globals[i] for i in range(len(globals)) if global_feature_mask[0][i]==1]
            else:
                global_ptm_papers = globals
            
            if local_feature_mask is not None:
                local_ptm_papers = [locals[i] for i in range(len(locals)) if local_feature_mask[0][i]==1]
            else:
                local_ptm_papers = locals
            ptm_papers = global_ptm_papers+local_ptm_papers

            if self.model_args.use_oagbert:  #denotes using oagbert to encoding paper authors
                text_inputs = []
                for p in ptm_papers: #ptm_papers[0]
                    title, authors, orgs, venue = self._get_oagbert_input(self.pub[p],self.author[author_id]['name'])
                    if "title" not in self.model_args.text_feature:
                        title = ''
                    if "author" not in self.model_args.text_feature:
                        authors = []
                    if "org" not in self.model_args.text_feature:
                        orgs = []
                    if "venue" not in self.model_args.text_feature:
                        venue = ''

                    if not (is_not_none(title) or is_not_none(authors) or is_not_none(orgs) or is_not_none(venue)):
                        authors = [self.author[author_id]['name']] 

                    text_inputs.append(self.ptm_tokenizer.build_inputs(title = title, authors = authors, affiliations = orgs, venue = venue, max_seq_length=512))

                text_inputs = self._to_max_length_tensors(text_inputs)
                # if text_inputs['input_ids'].shape[-1] == 0:
                #     breakpoint()
            else:
                if self.model_args.text_feature == "title":
                    ptm_papers_txt = [self.pub[p]['title'] for p in ptm_papers]
                else:
                    ptm_papers_txt = [self._get_paper_str(self.pub[p]) for p in ptm_papers]
                text_inputs = self.ptm_tokenizer(ptm_papers_txt, return_tensors='pt',add_special_tokens=True, truncation=True,padding=True,max_length=512)   
        return{
            'input_ids':llm_inputs['input_ids'],
            'attention_mask':llm_inputs['attention_mask'],
            'position_ids':torch.arange(llm_inputs['input_ids'].shape[-1]).unsqueeze(0),
            'text_inputs':text_inputs,
            "labels":self.data[index]['labels'] if self.mode == "train" else None,
            'author':self.data[index]['author'],    
            'pubs':self.data[index]['pubs'],
            'graph_emb': graph_emb
        }
    
    def _to_max_length_tensors(self,inputs):

        max_len = max([len(i[0]) for i in inputs])
        batch_input_ids = []
        batch_token_type_ids = []
        batch_attention_mask = []
        batch_position_ids = []
        batch_position_ids_second = []

        for i in inputs:
            input_ids, attention_mask, token_type_ids, _, position_ids, position_ids_second, _,_ = i
            batch_input_ids.append(input_ids+ [0]* (max_len-len(input_ids)))
            batch_attention_mask.append(attention_mask+ [0]*( max_len-len(attention_mask)))
            batch_token_type_ids.append(token_type_ids+ [0]* (max_len-len(token_type_ids)))
            batch_position_ids.append(position_ids+ [0]* (max_len-len(position_ids)))
            batch_position_ids_second.append(position_ids_second+ [0]* (max_len-len(position_ids_second)))
        res = {
            'input_ids':torch.tensor(batch_input_ids,dtype=torch.long),
            'attention_mask':torch.tensor(batch_attention_mask,dtype=torch.long),
            'token_type_ids':torch.tensor(batch_token_type_ids,dtype=torch.long),
            'position_ids':torch.tensor(batch_position_ids,dtype=torch.long),
            'position_ids_second':torch.tensor(batch_position_ids_second,dtype=torch.long)
        }
        return res
    
    def _generate_random_mask_context(self,mask,titles):
        res = []
        for i in range(len(titles)):
            info =""
            if mask[0][i] == 1:
                info += EMBED_TOKEN
            if mask[1][i] == 1:
                info += GRAPH_TOKEN
            if mask[2][i] == 1:
                info += titles[i]
                
            res.append(info)
        return res

    def _get_cut_num(self,globals_txt,length):
        tokenized_profile = [self.tokenizer.tokenize(i) for i in globals_txt]
        len_profile = [len(i)+2 for i in tokenized_profile] # +2 means to left space for graph emb and text emb
        sum_len = sum(len_profile)
        if sum_len < length:
            return len(globals_txt)
        
        total_len = 0
        p = 0   
        while total_len < length and p < sum_len:
            total_len += len_profile[p]
            p += 1
        return p-1

    def _get_oagbert_input(self,data,cur_author_name):
        """
        input: papaer info from pub file
        
        output: title author organization venue
        """
        title = ""
        authors = []
        orgs = []
        venue = []
        
        title = cut_to_target_len(data['title'],30)
        for author in data['authors'][:20]+ data['authors'][-1:]:
            name = author['name']
            if is_chinese(name):
                name = get_pin_yin(name)
            if not match(cur_author_name, cleaning_name(name)):
                authors.append(name)
            if is_not_none(author['org']):
                orgs.append(author['org']) 

        orgs = list(set(orgs))
        venue = cut_to_target_len(data['venue'],30)
        return title, authors, orgs, venue
        
        
    def _get_paper_str(self,data):
        res = '{ '
        if data['title'] != '':
            title = decoding(data['title'])
            res += f'Title: {title}  '
        if data['venue'] != '' and data['venue'] is not None:
            venue = decoding(' '.join(data['venue'].split(' ')[:20]))
            venue += f'  Venue: {venue} '
            res+= venue
        if len(data['authors']) != 0:
            authors= ' Author: '
            orgs = ' Organization: '
            for i in data['authors'][:10]:
                name,org = i['name'],i['org']
                if name != '':
                    authors +=  '{} ;'.format(get_pinyin(name))
                if org != '':
                    orgs += '{} ;'.format(decoding(org))
            res += authors+ orgs
        # if data['keywords'] != [] and data['keywords'] != '':
        #     keywords = decoding(','.join(data['keywords']))
        #     keywords= f'# Keywords: {keywords} '
        #     res += keywords
        res += ' }'
        res = res.lower()
        res = res.strip()
        return res
       
    def _get_author_str(self,data, cur_author):
        cur_author = cleaning_name(cur_author)

        res = []
        cuted_author = []
        #取data['authors']前20个和最后一个作者

        if len(data['authors'])>20:
            cuted_author = data['authors'][:20] +data['authors'][-1:]
        else:
            cuted_author = data['authors']
            
        for i in cuted_author:

            name = i['name']
            if is_chinese(name):
                name = get_pin_yin(name)
            if not match(cur_author, cleaning_name(name)):
                res.append(name)
        if res == []:  # 可能导致结果变差
            res = [cur_author]
        return res


funcs = [
    match_name_one,
    match_name_two,
    match_name_three,
    match_name_four,
    match_name_five,
    match_name_six,
    match_name_seven,
]
def match(name1,name2, loose = False):

    clean_name1 = name1
    clean_name2 = name2
    for f in funcs:
        if f(clean_name1, clean_name2, loose):
            return True
        if f(clean_name2, clean_name1, loose):
            return True
    return False

def cleaning_name(name):
    if is_chinese(name):
        name = get_pin_yin(name)
    name = unidecode(name)
    name = name.lower()
    new_name = ""
    for a in name:
        if a.isalpha():
            new_name += a
        else:
            new_name = new_name.strip()
            new_name += " "
    return new_name.strip()

@dataclass
class DataCollatorForPacking:
    def __call__(self, features):
        features = features[0]
        return {
            'input_ids':features['input_ids'],
            'attention_mask':features['attention_mask'],
            'position_ids':features['position_ids'],
            'text_inputs':features['text_inputs'],
            'labels':  features['labels'],
            'author':features['author'],
            'pubs':features['pubs'],
            'graph_emb':features['graph_emb']
        }

        # return{
        #     'input_ids':llm_inputs['input_ids'],
        #     'attention_mask':llm_inputs['attention_mask'],
        #     'position_ids':torch.arange(llm_inputs['input_ids'].shape[-1]).unsqueeze(0),
        #     'text_inputs':text_inputs if self.model_args.input_type else None,
        #     "labels":self.data[index]['labels'] if self.mode == "train" else None,
        #     'author':self.data[index]['author'],    
        #     'pubs':self.data[index]['pubs']
        # }


def generate_random_mask(length,prob):
    return np.random.choice([0,1],size=length,p=[1-prob,prob])

class INDPacking(Dataset): #for train data
    def __init__(self,dataset, tokenizer, data_args, model_args , mode = "train",ptm_tokenizer = None, use_graph = False, use_emb = False ):
        super(INDPacking, self).__init__()
        self.use_graph = use_graph
        self.use_emb = use_emb
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.model_args = model_args
        self.author, self.pub = dataset 
        self.ptm_tokenizer = ptm_tokenizer

        self.instruct = "Identify the abnormal text from the text collection according to the following rules:\n Here is a collection of paper titles: \n ### {} \n Determine whether the following is true and answer 'Yes' or 'No'. {}"

        self.local_instruct = "\"{}\" belongs to the main cluster. "+ LABEL_TOKEN + "." 
        self.instruct_length = len(self.tokenizer.tokenize(self.instruct+self.local_instruct))
        if use_graph:
            with open(self.model_args.graph_path, 'rb') as f:
                self.graph_emb = pickle.load(f)
        self.data = []
        self.mode = mode
        if self.mode == "train":
            if data_args.sample :
                for _ in range(4):
                    for key in self.author.keys():
                        if len(self.author[key]['outliers']) < len(self.author[key]['normal_data']): 
                            pos_set = random.sample(self.author[key]['normal_data'],len(self.author[key]['outliers']))
                            neg_set = self.author[key]['outliers']
                        elif len(self.author[key]['outliers']) > len(self.author[key]['normal_data']):
                            neg_set = random.sample(self.author[key]['outliers'],len(self.author[key]['normal_data']))
                            pos_set = self.author[key]['normal_data']
                        else:
                            neg_set = self.author[key]['outliers']
                            pos_set = self.author[key]['normal_data']
                        set = neg_set+pos_set
                        random.shuffle(set)
                        for i in range(len(set)//model_args.packing_size):
                            sampled_pack = set[i*model_args.packing_size:(i+1)*model_args.packing_size]
                            labels = [int(i in self.author[key]['normal_data']) for i in sampled_pack]
                            self.data.append({
                                "author": key,
                                "pubs":sampled_pack,
                                "labels":labels
                            })
            else:
                for key in self.author.keys():
                    set = self.author[key]['outliers']+self.author[key]['normal_data']
                    random.shuffle(set)
                    for i in range(len(set)//model_args.packing_size):
                        sampled_pack = set[i*model_args.packing_size:(i+1)*model_args.packing_size]
                        labels = [int(i in self.author[key]['normal_data']) for i in sampled_pack]
                        self.data.append({
                            "author": key,
                            "pubs":sampled_pack,
                            "labels":labels
                        })
            random.shuffle(self.data)
        else:
            for key in self.author.keys():
                if 'papers' in self.author[key]:
                    set = self.author[key]['papers']
                elif 'normal_data' in self.author[key]:
                    set = self.author[key]['normal_data']+self.author[key]['outliers']
                for i in range(0, len(set), model_args.packing_size):
                    sampled_pack = set[i:i + model_args.packing_size]
                    self.data.append({
                        "author": key,
                        "pubs":sampled_pack,
                    })
        if mode != "train" and data_args.sorted_file is not None:
            with open(data_args.sorted_file,'r') as f:
                self.sorted_file = json.load(f)
        else:
            self.sorted_file = None
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        author_id = self.data[index]['author']
        if self.mode == "train":
            normal_data = self.author[author_id]['normal_data']
            outliers = self.author[author_id]['outliers']
            if len(normal_data) / len(outliers) < 0.7 and self.data_args.normalize_profile:
                profile = normal_data + random.sample(outliers, min(len(normal_data),int(len(outliers)*0.5)))               
            else:
                profile = normal_data + outliers
            profile = self.author[self.data[index]['author']]['normal_data'] +self.author[self.data[index]['author']]['outliers']
        else:
            profile = self.author[self.data[index]['author']]['papers']

        if self.mode != "train" and self.sorted_file is not None:
            profile = sorted(profile, key=lambda x: self.sorted_file[self.data[index]['author']][x], reverse=True)
            profile = profile[:max(int(len(profile)*0.6),3)]
        else:
            random.shuffle(profile)
        # random drop papers
        if self.mode == "train" and self.model_args.papers_drop_ratio > 0:
            profile = random.sample(profile,int(len(profile) * (1 - self.model_args.papers_drop_ratio)))    
        globals = profile[:300] #if self.model_args.input_type == "text" else profile[:1000]


        locals = self.data[index]['pubs']

        globals_txt = [(GRAPH_TOKEN if self.use_graph else '')+ \
                      (EMBED_TOKEN if self.use_emb else '')+ \
                      (self.pub[p]['title']) for p in globals
                      ]
        locals_txt = [(GRAPH_TOKEN if self.use_graph else '')+ \
                      (EMBED_TOKEN if self.use_emb else '')+ \
                      (self.pub[p]['title']) for p in locals
                      ]
        
        local_instruct = "# ".join([self.local_instruct.format(i) for i in locals_txt])
        locals_length = len(self.tokenizer.tokenize(local_instruct))
        
        #cut global to target length
        cut_num = self._get_cut_num(globals_txt,self.data_args.max_source_length-1000-locals_length)
        globals_txt = globals_txt[:cut_num]
        globals = globals[:cut_num]
        global_instruct = '#'.join(globals_txt)
        
        context = self.instruct.format(global_instruct,local_instruct)
        llm_inputs = self.tokenizer(context, return_tensors='pt',add_special_tokens=True, truncation=True,max_length=self.data_args.max_source_length)

        graph_emb = None
        text_inputs = None
        if self.use_graph:
            graph_emb = [self.graph_emb[author_id][i] for i in globals +locals]
            graph_emb = torch.tensor(graph_emb,dtype=torch.float32)
            graph_emb = torch.cat([graph_emb,torch.tensor(self.graph_emb[author_id]['graph'],dtype=torch.float32).expand(graph_emb.shape[0],-1)],dim=-1) #to debuged
        if self.use_emb:
            ptm_papers = globals+locals
            if self.model_args.text_feature == "title":
                ptm_papers_txt = [self.pub[p]['title'] for p in ptm_papers]
            else:
                ptm_papers_txt = [self._get_paper_str(self.pub[p]) for p in ptm_papers]
            text_inputs = self.ptm_tokenizer(ptm_papers_txt, return_tensors='pt',add_special_tokens=True, truncation=True,padding=True,max_length=512)

        return{
            'input_ids':llm_inputs['input_ids'],
            'attention_mask':llm_inputs['attention_mask'],
            'position_ids':torch.arange(llm_inputs['input_ids'].shape[-1]).unsqueeze(0),
            'text_inputs':text_inputs,
            "labels":self.data[index]['labels'] if self.mode == "train" else None,
            'author':self.data[index]['author'],    
            'pubs':self.data[index]['pubs'],
            'graph_emb': graph_emb
        }

    def _get_cut_num(self,globals_txt,length):
        tokenized_profile = [self.tokenizer.tokenize(i) for i in globals_txt]
        len_profile = [len(i) for i in tokenized_profile]
        sum_len = sum(len_profile)
        if sum_len < length:
            return len(globals_txt)
        
        total_len = 0
        p = 0   
        while total_len < length and p < sum_len:
            total_len += len_profile[p]
            p += 1
        return p-1
    
    def _get_paper_str(self,data):
        res = '{ '
        if data['title'] != '':
            title = decoding(data['title'])
            res += f'Title: {title}  '
        if data['venue'] != '' and data['venue'] is not None:
            venue = decoding(' '.join(data['venue'].split(' ')[:20]))
            venue += f'  Venue: {venue} '
            res+= venue
        if len(data['authors']) != 0:
            authors= ' Author: '
            orgs = ' Organization: '
            for i in data['authors'][:10]:
                name,org = i['name'],i['org']
                if name != '':
                    authors +=  '{} ;'.format(get_pinyin(name))
                if org != '':
                    orgs += '{} ;'.format(decoding(org))
            res += authors+ orgs
        # if data['keywords'] != [] and data['keywords'] != '':
        #     keywords = decoding(','.join(data['keywords']))
        #     keywords= f'# Keywords: {keywords} '
        #     res += keywords
        res += ' }'
        res = res.lower()
        res = res.strip()
        return res