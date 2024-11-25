from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import random
import pickle
import torch
import json
import os

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

from ko_mrc.datasets import CustomedDataset
from ko_mrc.utils import edit_distance
from tensorboardX import SummaryWriter


def get_parser():
    parser = argparse.ArgumentParser(description='Parameters to train net')
    ## Hyper Parameters
    parser.add_argument('--n_epochs', default=3, type=int, help='Epoch to train the network')
    parser.add_argument('--batch_size', default=8, type=int, help='Training batch size')
    parser.add_argument('--lr', default=5e-5, type=float, help='Learning Rate')
    parser.add_argument('--wd', default=0, type=float, help='Weight Decay for optimizer')
    ## Data Path
    parser.add_argument('--train_path', default="Your Path", type=str, help='The train data path')
    parser.add_argument('--valid_path', default="", type=str, help='The valid data path')
    ## Training Resource Setting
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda')
    parser.add_argument('--device', default=0, type=int, help='GPU id')
    ## Save & Load
    parser.add_argument('--pretrained_model', default='', type=str, help='Pretrained base model')
    parser.add_argument('--save_path', default='weights', type=str, help='Location to save checkpoint models')
    parser.add_argument('--save_prob', default=True, type=bool, help='Save Probability')
    args = parser.parse_args()
    return args


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = True  


def read_data(path, mode='Train'):
    with open(path, 'rb') as f:
        data_dict = json.load(f)
    contexts = []
    questions = []
    answers = []
    for group in tqdm(data_dict['data']):
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
            
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)      
    return contexts, questions, answers


def read_test_data(path):
    with open(path, 'rb') as f:
        data_dict = json.load(f)
    contexts = []
    questions = []
    guids = []
    for group in tqdm(data_dict['data']):
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                guid = qa['guid']
                contexts.append(context)
                questions.append(question)
                guids.append(guid)
    return contexts, questions, guids     


def add_end_idx(contexts, answers):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2
    return contexts, answers


def train(model, optimizer, scheduler, loader, epochs, device, writer, args):
    print("TRAIN START")
    for ep in range(epochs):
        total_metric = {"Loss": 0, "Acc": 0}
        model.to(device)
        model.train()
        for idx, batch in tqdm(enumerate(loader), 
                               desc=f"Train {ep} / {epochs}", 
                               total=len(loader)):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions)
            
            start_logits, end_logits = outputs.start_logits, outputs.end_logits
            start_index, end_index = start_logits.argmax(dim=-1), end_logits.argmax(dim=-1)
            
            correct_cnt = (start_index == start_positions).sum() + (end_index == end_positions).sum()
            acc = correct_cnt / len(input_ids)
            loss = outputs.loss
            loss.backward()           
            
            optimizer.step()
            scheduler.step()
            
            total_metric["Loss"] = loss.item()
            total_metric["Acc"] = acc.item()
            if idx % 100 == 0:
                step = len(loader) * ep + idx
                writer.add_scalar("Train/Loss_Iter", loss.item(), step)
                writer.add_scalar("Train/Acc_Iter", acc, step)
        writer.add_scalar("Train/Loss_Avg", total_metric["Loss"] / idx, step)
        writer.add_scalar("Train/Acc_Avg", total_metric["Acc"] / idx, step)
        
        torch.save(model.state_dict(), f"result/{args.save_path}/epoch_{ep}.pth")
    print("TRAIN END")


def prediction(model, tokenizer, contexts, questions, guids, device, args):
    model.to(device)
    model.eval()
    
    result = []
    start_probs, end_probs = None, None
    with torch.no_grad():
        for context, question, guid in tqdm(zip(contexts, questions, guids), 
                                            desc="Inference", 
                                            total=len(contexts)):
            encodings = tokenizer(context, 
                                  question, 
                                  max_length=512, 
                                  truncation=True,
                                  padding="max_length", 
                                  return_token_type_ids=False)
            encodings = {key: torch.tensor([val]) for key, val in encodings.items()}
            
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            start_logits, end_logits = outputs.start_logits, outputs.end_logits
            if args.save_prob:
                start_prob, end_prob = start_logits.softmax(dim=-1).cpu().numpy(), end_logits.softmax(dim=-1).cpu().numpy()
                if start_probs is None:
                    start_probs, end_probs = [start_prob], [end_prob]
                else:
                    start_probs.append(start_prob)
                    end_probs.append(end_prob)
            token_start_index, token_end_index = start_logits.argmax(dim=-1), end_logits.argmax(dim=-1)
            result.append([guid, token_start_index.cpu().item(), token_end_index.cpu().item()])
    if not start_probs is None:
        with open(f"result/{args.save_path}/start_prob.pickle", 'wb') as f:
            pickle.dump(start_probs, f, pickle.HIGHEST_PROTOCOL)
        with open(f"result/{args.save_path}/end_prob.pickle", 'wb') as f:
            pickle.dump(start_probs, f, pickle.HIGHEST_PROTOCOL)
    return result


def main(args):
    seed_everything()
    
    DATA_PATH = args.train_path
    EPOCHS = args.n_epochs
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch_size
    DEVICE = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() and args.cuda else torch.device('cpu')
    os.makedirs(f"result/{args.save_path}", exist_ok=True)
    writer = SummaryWriter(f"result/{args.save_path}/tb_logs")

    train_contexts, train_questions, train_answers = read_data(DATA_PATH + "train.json")
    train_contexts, train_answers = add_end_idx(train_contexts, train_answers)

    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    model = AutoModelForQuestionAnswering.from_pretrained("klue/bert-base")
    
    train_dataset = CustomedDataset(train_contexts, train_questions, train_answers, 512, tokenizer)
    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=BATCH_SIZE,
                                  num_workers=0,
                                  shuffle=True,
                                  )
    
    optimizer = AdamW(model.parameters(), 
                      lr=LEARNING_RATE, 
                      weight_decay=2e-5,
                      )
    
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=total_steps // 10, 
                                                num_training_steps=total_steps)

    
    train(model, optimizer, scheduler, train_dataloader, EPOCHS, DEVICE, writer, args)
    test_contexts, test_questions, test_guids = read_test_data(DATA_PATH + "test.json")
    pred_steds = prediction(model, tokenizer, test_contexts, test_questions, test_guids, DEVICE, args)
    
    predictions = []
    for pred_sted, context, question in tqdm(zip(pred_steds, test_contexts, test_questions),
                                             desc="make results",
                                             total=len(pred_steds)):
        position = 0
        text = context + '[SEP]' + question
        context_position =[]
        for morph in tokenizer.tokenize(text):
            morph_text_only = morph.replace('#','')
            position = context.find(morph_text_only, position)
            context_position.append((position, position + len(morph_text_only)))
            position += len(morph_text_only)
        
        guid = pred_sted[0]
        start = pred_sted[1] - 1
        end = pred_sted[2] - 1
        try:
            answer = context[context_position[start][0]: context_position[end][1]]
        except:
            answer = ""
        predictions.append((guid, answer))
    results = pd.DataFrame(predictions, columns = ['ID', 'Predicted'])
    results.to_csv(f'result/{args.save_path}/result.csv', index=False)
    estimated_score = edit_distance(results)
    with open(f'result/{args.save_path}/scroe.txt', 'w') as f:
        f.write(f"{estimated_score}")


if __name__ == "__main__":
    args = get_parser()
    main(args)