import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from model import Net
from data_load import NerDataset, pad, VOCAB, tokenizer, tag2idx, idx2tag
import os
import numpy as np
import argparse
import my_test 
from 处理函数 import *
import pandas as pd
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


def test(model, iterator, f):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch  #is_heads表示 该字是否为切词后的第一个字
            
            _, _, y_hat = model(x, y)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            #Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    ## gets results and save
    with open("temp.txt", 'w') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            assert len(preds)==len(words.split())==len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write("%s %s %s\n"%(w,t,p))
            fout.write("\n")
    ## calc metric
    y_true =  np.array([tag2idx[line.split()[1]] for line in open("temp.txt", 'r').read().splitlines() if len(line) > 0])
    y_pred =  np.array([tag2idx[line.split()[2]] for line in open("temp.txt", 'r').read().splitlines() if len(line) > 0])

    # num_proposed = len(y_pred[y_pred>1])
    # num_correct = (np.logical_and(y_true==y_pred, y_true>1)).astype(np.int).sum()
    # num_gold = len(y_true[y_true>1])
    
    # print(f"num_proposed:{num_proposed}")
    # print(f"num_correct:{num_correct}")
    # print(f"num_gold:{num_gold}")
    # try:
    #     precision = num_correct / num_proposed
    # except ZeroDivisionError:
    #     precision = 1.0

    # try:
    #     recall = num_correct / num_gold
    # except ZeroDivisionError:
    #     recall = 1.0

    # try:
    #     f1 = 2*precision*recall / (precision + recall)
    # except ZeroDivisionError:
    #     if precision*recall==0:
    #         f1=1.0
    #     else:
    #         f1=0

    #final='../data/ner_result.txt'
    #final = f + ".P%.4f_R%.4f_F%.4f" %(precision, recall, f1)
    #with open(final, 'w') as fout:
        #result = open("temp", "r").read()
        #fout.write(f"{result}\n")

        #fout.write(f"precision={precision}\n")
        #fout.write(f"recall={recall}\n")
        #fout.write(f"f1={f1}\n")

    #os.remove("temp")

    # print("precision=%.4f"%precision)
    # print("recall=%.4f"%recall)
    # print("f1=%.4f"%f1)
    
    import csv
    ###创建pipeline中关系抽取测试集
    path='./data/ner_result.tsv'
    with open(path, 'w') as fout:
        result = open("temp.txt", "r").read().strip().split("\n\n")

        wtest=csv.writer(fout,delimiter='\t')
        for ID,entry in enumerate(result):
            entities=[]
            entity={}
            words = [line.split()[0] for line in entry.splitlines()]
            tags = ([line.split()[-1] for line in entry.splitlines()])
    #         print(words,tags)
            for idx,t in enumerate(tags):
    #             print(t)
                if t=='O':
                    if entity!={}:
                        entity['end']=idx
                        entities.append(entity)
                        entity={}   #type start end text
                    continue
                elif t.startswith('B'): 
                    entity['start']=idx
                    entity['text']=words[idx]
                    entity['type']=t.split('-')[1]
                elif t.startswith('I'):
                    if'text' in entity:
                        entity['text']+=words[idx]

            entitypair_list=[]
            for e1 in entities:
                if e1['type']!='Nh' and e1['type']!='NDR':
                    continue
                for e2 in entities:
                    if e2['type']!='Nh' and e2['type']!='NDR':
                        continue
                    if e1!=e2 and e1['text']!=e2['text']:
                        entitypair=[]
                        entitypair.append(e1)
                        entitypair.append(e2)
                        if entitypair not in entitypair_list:
                            entitypair_list.append(entitypair)

            for sid,ep in enumerate(entitypair_list):
                sent=words.copy()
                e1start=ep[0]['start']
                e1end=ep[0]['end']
                e2start=ep[1]['start']
                e2end=ep[1]['end']

                if e1end>400 or e2end>400:
                    continue

                if e1start<e2start:
                    sent.insert(e2end,'[E22]')
                    sent.insert(e2start,'[E21]')
                    sent.insert(e1end,'[E12]')
                    sent.insert(e1start,'[E11]')
                else:
                    sent.insert(e1end,'[E12]')
                    sent.insert(e1start,'[E11]')
                    sent.insert(e2end,'[E22]')
                    sent.insert(e2start,'[E21]')

                wtest.writerow((sid,''.join(sent),0,ID))

    # return precision, recall, f1

    
def file_trans(path):
    with open(path, 'w') as fout:
        result = open("temp", "r").read()

# def write_result(dic_triple,path):
#     with open(path,'w',encoding='utf-8')as fw:
#         for key in dic_triple:
#             dic_triple[key]
#         fw.writer.write("%s\t%s\t%s\t%s\n" %(lines[key][0], lines[key][1],str(RELATION_LABELS[preds[key]]),lines[key][3]))
if __name__=="__main__":

    path=r'./筛选1500'
    pathDir =os.listdir(path)
    print("=========start=========")
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--top_rnns", dest="top_rnns", action="store_true")
    parser.add_argument("--logdir", type=str, default="my_result")
    parser.add_argument("--testset", type=str, default="ner_txt.txt")
    
    hp = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=========build model=========")
    model = Net(hp.top_rnns, len(VOCAB), device, hp.finetuning).cuda()
    # model=1
    print("=========parallel=========")
    model = nn.DataParallel(model)
    
    df_case=pd.read_csv('df_case.csv')
    # make_ner_txt(df_case.inf,hp.testset)   

    print("=========load data=========")

    test_dataset = NerDataset(hp.testset)
                                   
    test_iter = data.DataLoader(dataset=test_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr = hp.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    # f1_best=0
    # fname = os.path.join(hp.logdir, 'model')
    
    # print("=========ner=========")
    # test_model=Net(hp.top_rnns, len(VOCAB), device, hp.finetuning)
    # test_model = nn.DataParallel(test_model)
    # test_model.load_state_dict(torch.load("%s.pt"%(fname)))
    # test_model.cuda()
    # test(test_model, test_iter, fname)

    print("=========re=========")
    dic_triple=my_test.main()
    # print(len(dic_triple))
#     dic_triple=rec_weight(dic_triple)
#     for i in range(len(dic_triple)):
# #     print(i)

#         df_case.loc[i,'triples']='|'.join(list(dic_triple[str(i)]))

#     df_case.to_csv('df_case_triples.csv',encoding='utf-8',sep=',',index=False)
#     print("=========finish=========")
    

