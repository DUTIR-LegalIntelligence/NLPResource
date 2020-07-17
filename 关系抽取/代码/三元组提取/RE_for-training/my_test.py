import argparse
import glob
import logging
import os
import sys
import random
import torch.nn as nn
import numpy as np
import torch
import socket
import csv
os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
# import ptvsd
# Allow other computers to attach to ptvsd at this IP address and port.
# ptvsd.enable_attach(address=('192.168.11.2', 3000), redirect_output=True)
# Pause the program until a remote debugger is attached
# ptvsd.wait_for_attach()


from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
# from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from torch.nn import MSELoss, CrossEntropyLoss
from pytorch_transformers import (
    WEIGHTS_NAME, BertConfig, BertModel, BertPreTrainedModel, BertTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule
import torch.nn.functional as F
from my_utils import (RELATION_LABELS, compute_metrics, convert_examples_to_features,
                   output_modes, data_processors)
from my_bert_better import BertForSequenceClassification
import json
from argparse import ArgumentParser
from my_config import Config
import re
logger = logging.getLogger(__name__)
additional_special_tokens = ["[E11]", "[E12]", "[E21]", "[E22]"]

pat_e1=r'\[E11\](.*?)\[E12\]'
pat_e2=r'\[E21\](.*?)\[E22\]'
'''
class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode(
            "Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.l2_reg_lambda = config.l2_reg_lambda
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(
            config.hidden_size*3, self.config.num_labels)
        self.tanh = nn.Tanh()
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, e1_mask=None, e2_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # for details, see the document of pytorch-transformer
        pooled_output = outputs[1]
        sequence_output = outputs[0]
        #print(sequence_output.shape,sequence_output[-1].shape)
        # mask method 1
        # p.s. This makes me crazy that the mask method does not work, it should work...
        # extended_e1_mask = e1_mask.unsqueeze(2)

        # extended_e1_mask = (1.0 - extended_e1_mask) * -10000.0
        # extended_e1_mask = extended_e1_mask.expand_as(sequence_output)

        # extended_e2_mask = e2_mask.unsqueeze(2)
        # extended_e2_mask = (1.0 - extended_e2_mask) * -10000.0
        # extended_e2_mask = extended_e2_mask.expand_as(sequence_output)

        # e1 = self.tanh(sequence_output-extended_e1_mask.float()).mean(dim=1)
        # e2 = self.tanh(sequence_output-extended_e2_mask.float()).mean(dim=1)

        # mask method 2, it is not necessary to divide the length of entity.
        # the simplified verison even outperforms the full version.
        # e1_len = e1_mask.ne(0).float().sum(dim=1)
        # batch_size = e1_len.size(0)
        # for i in range(batch_size):
        #    e1_mask[i, :] /= e1_len[i]
        extended_e1_mask = e1_mask.unsqueeze(1)
        extended_e1_mask = torch.bmm(
            extended_e1_mask.float(), sequence_output).squeeze(1)

        # e2_len = e2_mask.ne(0).float().sum(dim=1)
        # batch_size = e2_len.size(0)
        # for i in range(batch_size):
        #    e2_mask[i, :] /= e2_len[i]
        extended_e2_mask = e2_mask.unsqueeze(1)
        extended_e2_mask = torch.bmm(
            extended_e2_mask.float(), sequence_output).squeeze(1)

        e1 = self.tanh(extended_e1_mask.float())
        e2 = self.tanh(extended_e2_mask.float())

        pooled_output = self.dropout(
            torch.cat([self.tanh(pooled_output), e1, e2], dim=-1))
        # print(pooled_output.size())
        logits = self.classifier(pooled_output)
        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]
        # probabilities = F.softmax(logits, dim=-1)
        # log_probs = F.log_softmax(logits, dim=-1)

        # one_hot_labels = F.one_hot(labels, num_classes=self.num_labels)

        # per_example_loss = - \
        #     tf.reduce_min(one_hot_labels[:, 1:] * log_probs[:, 1:], axis=-1)

        # l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

        # rc_probabilities = probabilities - probabilities * one_hot_labels
        # second_pre = - tf.reduce_max(rc_probabilities[:, 1:], axis=-1) + 1
        # # + tf.math.log(second_pre) * log_probs[:,0]
        # rc_loss = - tf.math.log(second_pre)

        # loss = tf.reduce_sum(per_example_loss) + 5 * \
        #     tf.reduce_sum(rc_loss) + l2 * l2_reg_lambda
        device = logits.get_device()
        loss = torch.sum(torch.tensor([torch.sum(p ** 2) / 2
                                       for p in self.parameters() if p.requires_grad])).to(device)*self.l2_reg_lambda
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss += loss_fct(logits.view(-1), labels.view(-1))
            else:
                # maybe next time I will add the L2 loss
                loss_fct = CrossEntropyLoss()
                loss += loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        
        return outputs  # (loss), logits, (hidden_states), (attentions)

'''
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def triple_evalute(dic_pred):
    dic_true={}
    with open('../data/test_triple.json','rt',encoding='utf-8')as fin:
    
        for line in fin:
            line = line.strip()
            if not line:
                continue  #结束则跳出循环
            dic_true= json.loads(line)#将json数据转换成python对象  这里是转换为了字典
    sum_pred=0.000000001
    sum_true=0.000000001
    sum_tp=0.000000001
    for key in dic_true:
        sum_true+=len(dic_true[key])
    for key in dic_pred:
        sum_pred+=len(dic_pred[key])
        if key in dic_true:
            for t in dic_pred[key]:
                t=list(t)
                if t in dic_true[key]:
                    sum_tp+=1
    
    precision=float(sum_tp/sum_pred)
    recall=float(sum_tp/sum_true)
    f1=float(2*precision*recall/(precision+recall))
    return precision,recall,f1

def evaluate(config, model, tokenizer, prefix=""):
    
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    with open('../data/ner_result.tsv', "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="\t")
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(cell for cell in line)
            lines.append(line)
    
    eval_task = config.task_name
    eval_output_dir = config.output_dir

    results = {}
    eval_dataset = load_and_cache_examples(
        config, eval_task, tokenizer, evaluate=True)
    
    if not os.path.exists(eval_output_dir) and config.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    config.eval_batch_size = config.per_gpu_eval_batch_size * \
        max(1, config.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(
        eval_dataset) if config.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=config.eval_batch_size)

    # Eval
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", config.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(config.device) for t in batch)
        #print(len(batch))
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      # XLM and RoBERTa don't use segment_ids
                      'token_type_ids': batch[2],
                      'labels':      batch[3],
                      'e1_mask': batch[4],
                      'e2_mask': batch[5],
                      
                      }
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            #print(np.argmax(logits.detach().cpu().numpy()))
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
            ID_list=batch[6].detach().cpu().numpy()
           
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            ID_list=np.append(ID_list,batch[6].detach().cpu().numpy())
        #print('input_ids',batch[0])
        #print('e1_mask', batch[4])
        #print('e2_mask',batch[5])
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    print(ID_list.shape,preds.shape)
    #result = compute_metrics(eval_task, preds, out_label_ids)
    #results.update(result)
    #print(results)
    #logger.info("***** Eval results {} *****".format(prefix))
    #for key in sorted(result.keys()):
        #logger.info("  %s = %s", key, str(result[key]))
    dic_triple={}
    output_eval_file = "eval/sem_res.txt"
    with open(output_eval_file, "w") as writer:
        for key in range(len(preds)):
            if lines[key][3] not in dic_triple:
                dic_triple[lines[key][3]]=set()
            if preds[key]==4:
                continue
            e1=re.search(pat_e1,lines[key][1]).group(1)
            e2=re.search(pat_e2,lines[key][1]).group(1)
            triple=tuple(([e1,str(RELATION_LABELS[preds[key]]),e2]))
            dic_triple[lines[key][3]].add(triple)
            writer.write("%s\t%s\t%s\t%s\n" %(lines[key][0], lines[key][1],str(RELATION_LABELS[preds[key]]),lines[key][3]))
            #writer.write("%d\t%s\t%s\n" %(key+8001, str(RELATION_LABELS[preds[key]]),str(RELATION_LABELS[out_label_ids[key]])))
    #print(dic_triple)
    p,r,f1=triple_evalute(dic_triple)
    print('P:',p,'R:',r,'F1:',f1)
    return dic_triple


def load_and_cache_examples(config, task, tokenizer, evaluate=False):
    if config.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    processor = data_processors[config.task_name]()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(config.data_dir, 'cached_{}_{}_{}_{}'.format(
        'pipe' ,
        list(filter(None, 'bert-base-uncased'.split('/'))).pop(),
        str(config.max_seq_len),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s",
                    config.data_dir)
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(
            config.data_dir) 
                                                                           
        #print(examples[0])
        
        features = convert_examples_to_features(
            examples, label_list, config.max_seq_len, tokenizer, "classification", use_entity_indicator=config.use_entity_indicator)
    if config.local_rank in [-1, 0]:
        logger.info("Saving features into cached file %s",
                    cached_features_file)
        torch.save(features, cached_features_file)

    if config.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()
    output_mode = "classification"
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_e1_mask = torch.tensor(
        [f.e1_mask for f in features], dtype=torch.long)  # add e1 mask
    all_e2_mask = torch.tensor(
        [f.e2_mask for f in features], dtype=torch.long)  # add e2 mask
    sent_id = torch.tensor([int(f.ID) for f in features], dtype=torch.long)  # add id
    if output_mode == "classification":
        all_label_ids = torch.tensor(
            [f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor(
            [f.label_id for f in features], dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_input_mask,
                            all_segment_ids, all_label_ids, all_e1_mask, all_e2_mask,sent_id)
    return dataset


def main():
    parser = ArgumentParser(
        description="BERT for relation extraction (classification)")
    parser.add_argument('--config', dest='config')
    args = parser.parse_args()
    config = Config(args.config)

    if os.path.exists(config.output_dir) and os.listdir(config.output_dir) and config.train and not config.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(config.output_dir))

    # Setup CUDA, GPU & distributed training
    if config.local_rank == -1 or config.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
        config.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(config.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        config.n_gpu = 1
    config.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if config.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   config.local_rank, device, config.n_gpu, bool(config.local_rank != -1))

    # Set seed
    set_seed(config.seed)

    # Prepare GLUE task
    processor = data_processors["semeval"]()
    output_mode = output_modes["semeval"]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if config.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    # Make sure only the first process in distributed training will download model & vocab
    bertconfig = BertConfig.from_pretrained(
         "/home/chenyanguang/RE/bert-pipeline/Robera", num_labels=num_labels, finetuning_task=config.task_name)
    bertconfig.l2_reg_lambda = config.l2_reg_lambda
    tokenizer = BertTokenizer.from_pretrained(
         "/home/chenyanguang/RE/bert-pipeline/Robera", do_lower_case=True, additional_special_tokens=additional_special_tokens)
    model = BertForSequenceClassification.from_pretrained(
         "/home/chenyanguang/RE/bert-pipeline/Robera", config=bertconfig)

    if config.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(config.device)

    # logger.info("Training/evaluation parameters %s", config)

    

    
    # Evaluation
    #results = {}
    dic_triples={}
    if True:
        tokenizer = BertTokenizer.from_pretrained(
            config.output_dir, do_lower_case=True, additional_special_tokens=additional_special_tokens)
        checkpoints = [config.output_dir]
        
        if config.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(
                glob.glob(config.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(
                logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split(
                '-')[-1] if len(checkpoints) > 1 else ""
            model = BertForSequenceClassification.from_pretrained(checkpoint)
            model.to(config.device)
            dic_triples= evaluate(config, model, tokenizer, prefix=global_step)
            #result = evaluate(config, model, tokenizer, prefix=global_step)
            #result = dict((k + '_{}'.format(global_step), v)for k, v in result.items())
            #results.update(result)

    
    return dic_triples


if __name__ == "__main__":
    main()
