from typing import Dict, Any, Generator, Tuple, List
import torch
from torch.utils.data import Dataset


class CustomedDataset(Dataset):
    def __init__(self, contexts, questions, answers, model_max_position_embedings, tokenizer):
        self.tokenizer = tokenizer
        self.answers = answers
        self.questions = questions
        self.contexts = contexts
        self.model_max_position_embedings = model_max_position_embedings
        print("Tokenizing ...")
        self.encodings = self.tokenizer(self.contexts, 
                                        self.questions,
                                        max_length=512, 
                                        truncation=True,
                                        padding="max_length",
                                        return_token_type_ids=False)
        print("Done !!!")
        self.add_token_positions()
        
    def add_token_positions(self):
        start_positions = []
        end_positions = []
        for i in range(len(self.answers)):
            start_positions.append(self.encodings.char_to_token(i, self.answers[i]['answer_start']))
            end_positions.append(self.encodings.char_to_token(i, self.answers[i]['answer_end'] - 1))
         
            if start_positions[-1] is None:
                start_positions[-1] = self.model_max_position_embedings
            if end_positions[-1] is None:
                end_positions[-1] = self.model_max_position_embedings
        self.encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

    def get_data(self):
        return {"contexts":self.contexts, 'questions':self.questions, 'answers':self.answers}
    
    def get_encodings(self):
        return self.encodings
    
    def __getitem__(self, idx):
        return {key:torch.tensor(val[idx]) for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    
