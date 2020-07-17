import re

def judgementedit(defendantname,judgement):
    P1= r'.*有期徒刑(.+?)，'
    P2= r'.*拘役(.+?)，'
    P3=r'.*罚金(.+?元)(?:；|。|（)'
    P4=r'(死刑.*)，'
    P5=r'(无期徒刑.*)，'
    p1=re.compile(P1)
    p2=re.compile(P2)
    p3=re.compile(P3)
    p4=re.compile(P4)
    p5=re.compile(P5)
    pt=u'(?:(.+)年)?(?:(.+?)个?月)?(?:(.+)(?:日|天))?'
    tripleset=set()
    while True:
            if defendantname!=[] and judgement!=[]:
                lt=[0,0,0]#刑期lt=[年,月,日]
                dn=defendantname.pop().strip()
                x=judgement.pop()
                if len(dn)>20:
                    continue
                m1=p1.search(x)
                m2=p2.search(x)
                m3=p3.search(x)
                m4=p4.search(x)
                m5=p5.search(x)
                flag=0
                if m1:
                    mt=re.match(pt,m1.group(1))
                    if mt:
                        for i in range(3):
                            lt[i]+=C2NUM(mt.group(i+1))
                    print(str(lt))
                    sumtime=lt[0]*365+lt[1]*30+lt[2]
                    tri=(dn,'有期徒刑',sumtime)
                    tripleset.add(tri)
                    flag=1
                if m2:
                    mt=re.match(pt,m2.group(1))
                    if mt:
                        for i in range(3):
                            lt[i]+=C2NUM(mt.group(i+1))
                    #print(str(lt))
                    sumtime=lt[0]*365+lt[1]*30+lt[2]
                    tri=(dn,'拘役',sumtime)
                    tripleset.add(tri)
                    flag=1
                if m3:
                    tri=(dn,'罚金',str(C2NUM(m3.group(1))))
                    tripleset.add(tri)
                    flag=1
                if m4:
                    tri=(dn,'死刑','')
                    tripleset.add(tri)
                    flag=1
                if m5:
                    tri=(dn,'无期徒刑','')
                    tripleset.add(tri)
                    flag=1
                '''if flag==0:
                    tri=(dn,'判决结果',x,'','')
                    tripleset.add(tri)'''


            else:
                break
    return(tripleset)

import xml.dom.minidom as md
import os
import re
import pandas as pd
def readxml(path):
    dom= md.parse(path)
    root=dom.documentElement
    #print(root.nodeName)
    lawlist=[]
    defendantname=[]
    judgement=[]
    type=[]
    fileid=None
    time=None
    place=None
    prosecutor=None
    title=None
    crimename=None
    inf=''
    for child in root.childNodes:
        '''
        if child.nodeName=='path':
            for text in child.childNodes:
                a=re.match(r'.+#(\d+).txt',text.nodeValue.split('\\')[3]).group(1)
                print(a)
        '''
        if child.nodeName=='FILEID':
            for text in child.childNodes:
                fileid=text.nodeValue
        if child.nodeName=='ID':
            for text in child.childNodes:
                fid=text.nodeValue
        if child.nodeName=='title':
            for text in child.childNodes:
                title=text.nodeValue
        if child.nodeName=='place':
            for text in child.childNodes:
                place=text.nodeValue
        if child.nodeName=='time':
            for text in child.childNodes:
                time=text.nodeValue
        if child.nodeName=='prosecutor':
            for text in child.childNodes:
                prosecutor=text.nodeValue
        if child.nodeName=='type':
            for text in child.childNodes:
                type.append(text.nodeValue)
        if child.nodeName=='defendant':
            for defendant in child.childNodes:
                if defendant.nodeName=='name':
                    for text in defendant.childNodes:
                        #print(text)
                        defendantname.append(text.nodeValue)
            for case in child.childNodes:
                if case.nodeName=='case':
                    for node in case.childNodes:
                        if node.nodeName=='crimename':
                            for text in node.childNodes:
                                crimename=text.nodeValue
                        if node.nodeName=='judgement':
                            for text in node.childNodes:
                                judgement.append(text.nodeValue)
                        if node.nodeName=='laws':
                            for law in node.childNodes:
                                for text in law.childNodes:
                                    lawlist.append(text.nodeValue)
                        if node.nodeName=='inf':
                            for text in node.childNodes:
                                inf=text.nodeValue




    return fid,fileid,title,place,time,prosecutor,crimename,judgement,lawlist,inf,defendantname,type


chs_arabic_map = {u'零':0, u'一':1, u'二':2,u'两':2, u'三':3, u'四':4,
        u'五':5, u'六':6, u'七':7, u'八':8, u'九':9,
        u'十':10, u'百':100, u'千':10 ** 3, u'万':10 ** 4,
        u'〇':0, u'壹':1, u'贰':2, u'叁':3, u'肆':4,
        u'伍':5, u'陆':6, u'柒':7, u'捌':8, u'玖':9,
        u'拾':10, u'佰':100, u'仟':10 ** 3, u'萬':10 ** 4,
        u'亿':10 ** 8, u'億':10 ** 8, u'幺': 1,
        u'０':0, u'１':1, u'２':2, u'３':3, u'４':4,
        u'５':5, u'６':6, u'７':7, u'８':8, u'９':9,
        u'0':0, u'1':1, u'2':2, u'3':3, u'4':4,
        u'5':5, u'6':6, u'7':7, u'8':8, u'9':9}

def C2NUM(chinese_digits):
    if chinese_digits is None:
        return 0
    if isinstance (chinese_digits, str):
        chinese_digits = chinese_digits
    result  = 0
    tmp     = 0
    hnd_mln = 0
    for count in range(len(chinese_digits)):
        curr_char  = chinese_digits[count]
        if curr_char in chs_arabic_map:
            curr_digit = chs_arabic_map.get(curr_char, None)
        else:
            continue
        # meet 「亿」 or 「億」
        if curr_digit == 10 ** 8:
            result  = result + tmp
            result  = result * curr_digit
            # get result before 「亿」 and store it into hnd_mln
            # reset `result`
            hnd_mln = hnd_mln * 10 ** 8 + result
            result  = 0
            tmp     = 0
        # meet 「万」 or 「萬」
        elif curr_digit == 10 ** 4:
            result = result + tmp
            result = result * curr_digit
            tmp    = 0
        # meet 「十」, 「百」, 「千」 or their traditional version
        elif curr_digit >= 10:
            tmp    = 1 if tmp == 0 else tmp
            result = result + curr_digit * tmp
            tmp    = 0
        # meet single digit
        elif curr_digit is not None:
            tmp = tmp * 10 + curr_digit
        else:
            return result
    result = result + tmp
    result = result + hnd_mln
    return result

def merge_xml(path,outpath):
    df=pd.DataFrame()
    pathDir =os.listdir(path)
    for p in pathDir:
        file = os.path.join('%s/%s' % (path, p))  # os.path.join()：  将多个路径组合后返回

        fid,fileid,title,place,time,prosecutor,crimename,judgement,lawlist,inf,defendantname,crimetype=readxml(file)
        inf=inf.strip().replace('\n','').replace('	','').strip().strip('"')
        p_dic={'fid':fid,'fileid':fileid,'title':title,'place':place,'time':time,'prosecutor':prosecutor,\
              'crimename':crimename,'judgement':'\t'.join(judgement),'lawlist':'\t'.join(lawlist),'inf':inf,\
              'defendantname':'\t'.join(defendantname),'crimetype':'\t'.join(crimetype)}
#         print(p_dic)
        new=pd.DataFrame(p_dic,index=[0])
        
        df=df.append(new,ignore_index=True)
#         print(p_dic)
#         print(fid,fileid,title,place,time,prosecutor,crimename,judgement,lawlist,inf,defendantname,crimetype)
    df.to_csv(outpath,encoding='utf-8',index=False)
    return df

def rec_weight(dic_triple):
    
    result = open("temp.txt", "r").read().strip().split("\n\n")

    for ID,entry in enumerate(result):
        flag_r=0
        for tri in dic_triple[str(ID)]:
            if tri[1]=='traffic_in' or tri[1]=='sell_drugs_to' or tri[1]=='provide_shelter_for':
                flag_r=1
                break
        if flag_r==1:
            continue
        
        entities=[]
        entity={}
        words = [line.split()[0] for line in entry.splitlines()]
        tags = ([line.split()[-1] for line in entry.splitlines()])
        entities=[]
        entity={}
        import numpy as np
        for idx,t in enumerate(tags):
            if t=='O':
                if entity!={}:
                    entity['end']=idx
                    if entity['type']=='NW'or entity['type']=='NDR':
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
        entity_p=set()
        for  e in entities:
            if e['type']=='NW':
                # print(e)
                span=100
                drug=None
                for d in entities:
                    
                    if d['type']=='NDR':
                        
                        if np.abs(d['start']-e['start'])<span:
                            drug=d
                            span=np.abs(d['start']-e['start'])      
                if drug!=None:
                    entity_p.add((drug['text'],'weight',e['text'])) 
        entity_d={}
        for p in entity_p:
            drug,_,weight=p
            if drug not in entity_d:
                entity_d[drug]=0
                
            if re.match(r"\d+\.?\d*",weight):
                entity_d[drug]+=float(re.match(r"\d+\.?\d*",weight).group(0))
            else:
                weight=str(C2NUM(weight))
#                 print(weight)
                entity_d[drug]+=float(re.match(r"\d+\.?\d*",weight).group(0))
        
        for ep in entity_d:
            dic_triple[str(ID)].add(str((ep,'weight',str(entity_d[ep])+'克'))  )
        # dic_triple[str(ID)]=tuple(dic_triple[str(ID)])          
    return dic_triple
# def rec_weight(inputStr):
#     '''                    
#     遍历tri_in_one_df中分词列，识别重量类型实体                    
#     规则:1.flag代表词列表词的数量（即序号），如果flag大于词表数量结束循环                    
#         2.判断某词是否在重量单位词典中同时n-1个词是否为量词，如果为真，                
#           将重量单位标记为E-NW，并进行下一步判断，如果为假，flag加1，对下一个词执行循环。                
#         3.判断货币单位n-2词的词性是否为量词，如果为真，将n-1词标注为I-NW，并循环判断之前词                
#           如果为假将n-1标记为B-NW，结束循环，flag加1，对下一个词执行循环                   
#     '''     

#     modelPath = './ltp_data_v3.4.0/'
#     seg= Segmentor()  # 生成对象
#     seg.load(modelPath + '/cws.model')
#     pos=Postagger()
#     pos.load_with_lexicon(modelPath + 'pos.model',modelPath + 'pos_lexicon2.txt')
#     segresult= seg.segment(inputStr)
#     #print(list(segresult))
#     posresult=pos.postag(segresult)
#     cut_list=list(segresult)
#     pos_list=list(posresult)

#     weight_dict= [line.strip() for line in open(dict_path +'重量单位词典.txt',encoding = 'utf-8-sig')]
#     try:                    
#         flag = 1                
#         while(flag < len(cut_list)):                
#             if (cut_list[flag] in weight_dict) and (pos_list[flag-1] == 'm'):            
#                 ner_list[flag] = 'E-NW'        
#                 x = True        
#                 i = 1        
#                 while(x):        
#                     if ((flag - (i + 1)) >= 0) and (pos_list[flag - (i + 1)] == 'm'):    
#                         ner_list[flag -i ] = 'I-NW'
#                         i += 1
#                     else:    
#                         ner_list[flag-i] = 'B-NW'
#                         x = False
#                 flag += 1        
#             else:            
#                 flag += 1        
#                 continue        
#     except Exception as e:                    
#         print(e)                
#         print(flag)                
                        
#     return ner_list

def make_ner_txt(sent_list,sent_path):
    with open(sent_path,'w',encoding='utf-8') as fout1:
        for sent in sent_list:
            sent =sent.replace('\n','')
            label=['O' for i in range(len(sentence_text))]
    #         print(sentence_text[112])

            for j in range(len(label)):
                fout1.write(sentence_text[j]+'\t'+label[j]+'\n')
            fout1.write('\n')
        
def make_re_data(ner_path):
    import csv
    ###创建pipeline中关系抽取测试集
    path='./data/ner_result.tsv'
    with open(path, 'w') as fout:
        result = open(ner_path, "r").read().strip().split("\n\n")

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
