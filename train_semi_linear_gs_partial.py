# Adapted from https://github.com/huggingface/transformers/blob/21da895013a95e60df645b7d6b95f4a38f604759/examples/run_glue.py
# for training GPT-2 medium for sequence classification with GeDi objective


# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import sys, csv, math


from modeling_gpt2 import GPT2LMHeadSemiPrefixLinearGSPartialModel, GPT2LMHeadPrefixModel, \
    GPT2LMHeadSemiPrefixLinearGSSepModel, GPT2LMHeadSemiPrefixLinearGSPartialModel_newenc

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel
)

# https://github.com/huggingface/transformers/blob/master/src/transformers/data/metrics/__init__.py
def acc_and_f1(preds, labels):
    assert len(preds) == len(labels)
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }
from sklearn.metrics import matthews_corrcoef, f1_score
def simple_accuracy(preds, labels):
        return (preds == labels).mean()

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadSemiPrefixLinearGSPartialModel, GPT2Tokenizer)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    for param in model.transformer.parameters():
        param.requires_grad=False
    for param in model.lm_head.parameters():
        param.requires_grad=False
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )


    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    temp=args.start_temp
    kl_weight=args.start_kl_weight
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    tr_sup_disc_loss, logging_sup_disc_loss = 0.0, 0.0
    tr_gen_loss, logging_gen_loss = 0.0, 0.0
    tr_kl_loss, logging_kl_loss = 0.0, 0.0
    tr_sup_encoder_loss, logging_sup_encoder_loss = 0.0, 0.0
    tr_sup_gen_loss, logging_sup_gen_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility


    for epoch_ in train_iterator:

        epoch_iterator = train_dataloader
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            batch_0 = batch[0]

            # #prepending tokens corresponding to 'positive' and 'negative' to the inputs
            # seq_a = (torch.ones(batch_0.shape[0])*pt_id).type_as(batch_0).view(-1,1)
            # seq_b = (torch.ones(batch_0.shape[0])*nt_id).type_as(batch_0).view(-1,1)


            seq_a = batch_0
            bsz = seq_a.shape[0]

            start_prefix_index=batch[4]
            end_prefix_index=batch[5]
            control_indexes=batch[6]
            task_list=batch[2]#bsz*task_num
            reverse_control_indexes = [random.choice(list(range(start_prefix_index[b_i],
                                                                control_indexes[b_i]))+
                                                     list(range(control_indexes[b_i]+1,
                                                                end_prefix_index[b_i]))) for b_i in range(bsz)]
            reverse_control_indexes=torch.LongTensor(reverse_control_indexes).unsqueeze(1).to(args.device)

            inputs={"input_ids": batch[0], 'attention_mask': None,"labels": batch[0],
                    'control_indexes':control_indexes.unsqueeze(1),
                    'reverse_control_indexes': reverse_control_indexes,
                    'task_list':task_list,
                    'task_mask': batch[3],
                    "temp": temp,
                    "do_semi_soft_one_hot": args.do_semi_soft_one_hot,
                    "no_gumbel": args.no_gumbel}


            outputs_agreeable = model(**inputs) #modeling_gpt2.py modified to have none reduction
            loss_semi_agreeable, loss_sup_agreeable,loss_sup_notagreeable, loss_kl, loss_encoder = outputs_agreeable[0]
            if args.n_gpu>1:
                loss_encoder=loss_encoder.mean()
                loss_kl=loss_kl.mean()
            loss_semi_agreeable=loss_semi_agreeable.view(bsz, -1)
            loss_sup_agreeable = loss_sup_agreeable.view(bsz, -1)
            loss_sup_notagreeable = loss_sup_notagreeable.view(bsz, -1)

            #loss mask includes first padded token

            loss_mask = batch[1][:,:-1].to(torch.float32).cuda()

            loss_lengths = torch.sum(loss_mask,1,keepdim=True)

            loss_semi_agreeable*=loss_mask
            loss_sup_agreeable*=loss_mask
            loss_sup_notagreeable*=loss_mask

            gen_loss = loss_semi_agreeable/loss_lengths
            gen_loss = torch.sum(gen_loss)/bsz

            sup_gen_loss = loss_sup_agreeable / loss_lengths
            sup_gen_loss = torch.sum(sup_gen_loss) / bsz

            if args.sum_loss:
                loss_sup_agreeable = loss_sup_agreeable.sum(dim=1)
                loss_sup_notagreeable= loss_sup_notagreeable.sum(dim=1)

            else:
                loss_sup_agreeable = (loss_sup_agreeable/loss_lengths).sum(dim=1)
                loss_sup_notagreeable= (loss_sup_notagreeable/loss_lengths).sum(dim=1)

            class_logits = torch.stack((-loss_sup_notagreeable, -loss_sup_agreeable), dim=1) #(bsz, 2) dimensional
            loss_fn = torch.nn.CrossEntropyLoss()
            ce_labels = batch[0].new_ones(bsz)
            sup_disc_loss = loss_fn(class_logits, ce_labels)

            if global_step< args.encoder_pretrain_step:
                loss=loss_encoder
            else:
                loss = sup_disc_loss*args.sup_disc_weight + args.gen_weight*gen_loss+loss_kl*kl_weight\
                       +args.sup_encoder_weight*loss_encoder+args.sup_gen_weight*sup_gen_loss


            # if np.isnan(loss.detach().cpu().numpy()):
            #     import pdb; pdb.set_trace()

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            tr_gen_loss += gen_loss.item()
            tr_kl_loss += loss_kl.item()
            tr_sup_disc_loss += sup_disc_loss.item()
            tr_sup_encoder_loss += loss_encoder.item()
            tr_sup_gen_loss += sup_gen_loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.temp_anneal_step>0 and global_step%args.temp_anneal_step==0:
                    temp = max(temp * math.exp(-args.temp_anneal_rate * global_step), args.min_temp)
                if args.kl_weight_anneal_step>0 and global_step%args.kl_weight_anneal_step==0:
                    kl_weight=min(args.max_kl_weight, kl_weight*math.exp(args.kl_weight_anneal_rate*global_step))

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer,global_step )
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value
                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    gen_loss_scalar = (tr_gen_loss - logging_gen_loss) / args.logging_steps
                    kl_loss_scalar = (tr_kl_loss - logging_kl_loss) / args.logging_steps
                    sup_disc_loss_scalar = (tr_sup_disc_loss - logging_sup_disc_loss)/args.logging_steps
                    sup_encoder_loss_scalar = (tr_sup_encoder_loss- logging_sup_encoder_loss)/args.logging_steps
                    sup_gen_loss_scalar = (tr_sup_gen_loss - logging_sup_gen_loss) / args.logging_steps

                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logs["gen_loss"]=gen_loss_scalar
                    logs["kl_loss"] =kl_loss_scalar
                    logs["temp"]=temp
                    logs["kl_weight"] = kl_weight
                    logs["sup_encoder_loss"] =sup_encoder_loss_scalar
                    logs["sup_disc_loss"] = sup_disc_loss_scalar
                    logs["sup_gen_loss"] = sup_gen_loss_scalar
                    logging_sup_disc_loss=tr_sup_disc_loss
                    logging_loss = tr_loss
                    logging_gen_loss = tr_gen_loss
                    logging_kl_loss = tr_kl_loss
                    logging_sup_encoder_loss = tr_sup_encoder_loss
                    logging_sup_gen_loss = tr_sup_gen_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    #torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    #torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    #logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, prefix_model,tokenizer, step, output_dir=None):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir if output_dir is None else output_dir
    # gen_model = model_class.from_pretrained(args.gen_model_name_or_path)
    # gen_model.to(args.device)
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    if not args.multi_eval:
        writer_list = []
        for writer_index in args.code_desired:
            writer = csv.writer(
                open(os.path.join(eval_output_dir, 'desired_' + str(writer_index) + '_' + str(step) + '_result.csv'),
                     'w'))
            writer.writerow(['id', 'comment_text', 'generated'])
            writer_list.append(writer)
    else:
        writer=csv.writer(
                open(os.path.join(eval_output_dir, 'desired_' + '_'.join([str(code) for code in args.code_desired]) + '_' + str(step) + '_result.csv'),
                     'w'))
        writer.writerow(['id', 'comment_text', 'generated'])

    results = {}

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    # multi-gpu eval
    # if args.n_gpu > 1 and not isinstance(prefix_model, torch.nn.DataParallel) and not isinstance(prefix_model,torch.nn.parallel.DistributedDataParallel):
    #     prefix_model = torch.nn.DataParallel(prefix_model)

    # Eval!
    logger.info("***** Running evaluation *****")

    #print(prefix_model.prefix_neg_theta_prime.data.size())
    with open(args.eval_file_path,'r') as f:
        reader=csv.DictReader(f)
        for row in reader:
            input_prompt = row['comment_text']
            id = row['id']
            text_ids = tokenizer.encode(input_prompt)
            encoded_prompts = torch.LongTensor(text_ids).unsqueeze(0).to(args.device)

            if not args.multi_eval:
                for wri, writer_index in enumerate(args.code_desired):
                    control_indexes = torch.LongTensor([int(writer_index)] * encoded_prompts.size(0)*args.gen_num_return_sequences).unsqueeze(
                        1).to(args.device)
                    generated_sequence_0 = prefix_model.generate(
                        input_ids=encoded_prompts,
                        max_length=args.gen_length,
                        temperature=args.gen_temperature,
                        top_k=args.gen_k,
                        top_p=args.gen_p,
                        repetition_penalty=args.gen_repetition_penalty,
                        eos_token_ids=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=args.gen_do_sample,
                        num_return_sequences=args.gen_num_return_sequences,
                        control_indexes=control_indexes,
                    )
                    generated_list = []
                    for i_generated in range(generated_sequence_0.size(0)):
                        text = tokenizer.decode(generated_sequence_0.tolist()[i_generated], clean_up_tokenization_spaces=True,
                                                skip_special_tokens=True)
                        generated_list.append(text)

                    writer_list[wri].writerow([id, input_prompt, json.dumps(generated_list)])
            else:
                control_indexes = torch.LongTensor(
                    [[int(code) for code in args.code_desired]] * encoded_prompts.size(0) * args.gen_num_return_sequences).to(args.device)
                generated_sequence_0 = prefix_model.generate(
                    input_ids=encoded_prompts,
                    max_length=args.gen_length,
                    temperature=args.gen_temperature,
                    top_k=args.gen_k,
                    top_p=args.gen_p,
                    repetition_penalty=args.gen_repetition_penalty,
                    eos_token_ids=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=args.gen_do_sample,
                    num_return_sequences=args.gen_num_return_sequences,
                    control_indexes=control_indexes,
                    prefix_num=len(args.code_desired)
                )
                generated_list = []
                for i_generated in range(generated_sequence_0.size(0)):
                    text = tokenizer.decode(generated_sequence_0.tolist()[i_generated],
                                            clean_up_tokenization_spaces=True,
                                            skip_special_tokens=True)
                    generated_list.append(text)
                writer.writerow([id, input_prompt, json.dumps(generated_list)])

    return results


def load_and_cache_examples(args, filepathlist, tokenizer, evaluate=False):
    assert(args.train_task_list[-1]=='topic')

    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    sup_data=[]
    prev_prefix_num=0
    if args.max_seq_length is None:
        max_length=tokenizer.max_len
    else:
        max_length=args.max_seq_length

    for taski, filepath in enumerate(args.train_file_path):
        data_taski={}
        with open(filepath, 'r') as f:
            reader=csv.DictReader(f)
            for row in reader:
                if not row['label'] in data_taski.keys():
                    data_taski[row['label']]=[]
                label_list=[-1]*len(args.train_task_list)
                label_list[taski]=int(row['label'])
                if 'imdb' in filepath.lower() and 'dbpedia' in args.train_file_path[-1].lower():
                    if 'amazon' in filepath.lower():
                        if row['source'].lower()=='imdb':
                            label_list[-1]=12
                    else:
                        label_list[-1]=12
                data_taski[row['label']].append([row['comment_text'], row['id'], label_list, int(row['label'])+prev_prefix_num])
        prev_prefix_num=sum(args.train_task_ncat_list[:(taski+1)])
        max_num_cat_data = 0
        for label in data_taski.keys():
            if len(data_taski[label]) > max_num_cat_data:
                max_num_cat_data = len(data_taski[label])
        for label in data_taski.keys():
            add_data = []
            if args.balanced:
                if args.sup_data_num > 0:
                    if len(data_taski[label]) > args.sup_data_num:
                        add_data = random.sample(data_taski[label], args.sup_data_num)
                    else:
                        add_data = (args.sup_data_num // len(data_taski[label])) * data_taski[label]
                        add_data += random.sample(data_taski[label], args.sup_data_num - len(add_data))
                else:
                    if len(data_taski[label]) < max_num_cat_data:
                        add_data = (max_num_cat_data // len(data_taski[label])) * data_taski[label]
                        add_data += random.sample(data_taski[label], max_num_cat_data - len(add_data))
                    else:
                        add_data = data_taski[label]

            else:
                if args.sup_data_num > 0:
                    if len(data_taski[label]) > args.sup_data_num:
                        add_data = random.sample(data_taski[label], args.sup_data_num)
                    else:
                        add_data = data_taski[label]
                else:
                    add_data = data_taski[label]
            for example in add_data:
                sup_data.append(example+[sum(args.train_task_ncat_list[:taski]), prev_prefix_num])
    sup_batch_encoding = tokenizer(
        [example[0] for example in sup_data],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_token_type_ids=True,
    )
    sup_input_ids = torch.tensor([sup_batch_encoding['input_ids'][i] for i in range(len(sup_data))], dtype=torch.long)
    sup_attention_mask = torch.tensor([sup_batch_encoding['attention_mask'][i] for i in range(len(sup_data))], dtype=torch.long)
    # sup_token_type_ids = torch.tensor([sup_batch_encoding['token_type_ids'][i] for i in range(len(sup_data))], dtype=torch.long)

    sup_labels = torch.tensor([example[2] for example in sup_data], dtype=torch.long)
    sup_prefix_idx=torch.tensor([example[3] for example in sup_data], dtype=torch.long)
    sup_task_masks=[]
    for example in sup_data:
        example_mask=[0]*example[4]+[1]*(example[5]-example[4])+[0]*(sum(args.train_task_ncat_list)-example[5])
        sup_task_masks.append(example_mask)
    sup_start_task_prefix_idx=torch.tensor([example[4] for example in sup_data], dtype=torch.long)
    sup_end_task_prefix_idx=torch.tensor([example[5] for example in sup_data], dtype=torch.long)
    sup_task_masks=torch.tensor(sup_task_masks, dtype=torch.long)
    sup_dataset=TensorDataset(sup_input_ids, sup_attention_mask, sup_labels,
                              sup_task_masks, sup_start_task_prefix_idx, sup_end_task_prefix_idx, sup_prefix_idx)
    return sup_dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_file_path",
        nargs='+',
        type=str,
        required=True,
        help="The input data path list. Should contain the .csv files",
    )
    parser.add_argument(
        "--train_task_list",
        nargs='+',
        type=str,
        required=True,
        help="The training task list. eg. toxicity sentiment",
    )
    parser.add_argument(
        "--train_task_ncat_list",
        nargs='+',
        type=int,
        required=True,
        help="The number of categories for each training task. eg. 2 2",
    )
    parser.add_argument(
        "--train_task_prefix_nfilter",
        nargs='+',
        type=int,
        help="The number of max selected prefix for each training task. eg. 1 1",
    )
    parser.add_argument(
        "--train_task_prefix_pfilter",
        nargs='+',
        type=float,
        help="The min prob of selected prefix for each training task. eg. 0.5 0.8",
    )
    parser.add_argument(
        '--eval_file_path',
        type=str,
        default=None,
        required=True,
        help='the evaluation data file path.'
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--eval_output_dir",
        default=None,
        type=str,
        help="The output directory where the completions will be written.",
    )
    parser.add_argument(
        "--output_sentiment_dir",
        default=None,
        type=str,
        help="The output directory where the sentiment model checkpoints is written.",
    )
    parser.add_argument(
        "--output_topic_dir",
        default=None,
        type=str,
        help="The output directory where the topic model checkpoints is written.",
    )
    parser.add_argument(
        "--output_encoder_dir",
        default=None,
        type=str,
        help="The output directory where the encoder model checkpoints is written.",
    )
    parser.add_argument(
        "--do_semi_soft_one_hot",
        action="store_true",
        help="Use semi soft one hot instead of complete soft one hot."
    )
    parser.add_argument(
        '--new_encoder',
        action="store_true",
        help="Use more complex encoder."
    )
    parser.add_argument(
        '--no_gumbel',
        action="store_true",
        help="No use of Gumbel softmax."
    )

    #new generative classifier specific parameters
    parser.add_argument("--balanced", action="store_true", help="use balanced dataset for training")
    parser.add_argument("--dropout",default=0.1,type=float, help="dropout prob")
    parser.add_argument("--gen_weight",default=0.0,type=float, help="scalar multiple for generative loss (lambda)")
    parser.add_argument("--disc_weight",default=0.0,type=float, help="scalar multiple for discriminative loss")
    parser.add_argument("--mask_rate", default=0.0, type=float, help="percentage of masked token for training")
    parser.add_argument("--sup_gen_weight", default=0.0, type=float, help="scalar multiple for supervised generative loss")
    parser.add_argument("--sup_encoder_weight", default=0.0, type=float, help="superivsed kl loss weight")
    parser.add_argument("--sup_disc_weight", default=0.0, type=float, help="supervised discriminative loss weight")

    parser.add_argument("--logit_scale",action="store_true",help="learns to scale logits for classification")
    parser.add_argument("--threeway", action="store_true", help="does 3-way classification")
    parser.add_argument("--sum_loss",action="store_true", help="sums losses")
    parser.add_argument("--outbias",action="store_true", help="learns output bias for each class")

    # Other parameters
    parser.add_argument('--sup_data_num', default=0, type=int, help='the number of supervised data for each prefix')

    parser.add_argument("--margin", default=0.5, type=float, help="the margin in the contrastive loss")
    parser.add_argument("--beta", default=0.25, type=float, help="dist loss weight")
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
            "--prefix_length",
            default=10,
            type=int,
            help="the length of the prefix."
    )
    parser.add_argument(
            "--prefix_hidden_size",
            default=800,
            type=int,
            help="the size of the prefix hidden size."
    )

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")




    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--start_temp", default=1.0, type=float, help="The initial temperature for gs.")
    parser.add_argument("--temp_anneal_rate", default=2e-7, type=float, help="The anneal rate for temperature.")
    parser.add_argument("--min_temp", default=0.5, type=float, help="minmum temperature.")
    parser.add_argument("--temp_anneal_step", default=100, type=int, help="temperature anneal step")

    parser.add_argument("--start_kl_weight", default=0.1, type=float, help="starting kl loss weight.")
    parser.add_argument("--kl_weight_anneal_rate", default=1e-6, type=float, help="the anneal rate for kl loss weight.")
    parser.add_argument("--max_kl_weight", default=3, type=float, help="max kl loss weight.")
    parser.add_argument("--kl_weight_anneal_step", default=100, type=int, help="kl loss weight anneal step.")

    parser.add_argument("--encoder_pretrain_step", type=int, default=0, help="the number of steps of pretraining encoder.")

    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    # parser.add_argument(
    #     "--fp16",
    #     action="store_true",
    #     help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    # )
    # parser.add_argument(
    #     "--fp16_opt_level",
    #     type=str,
    #     default="O1",
    #     help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    #     "See details at https://nvidia.github.io/apex/amp.html",
    # )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    # parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    # parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--mask_eos_token", action="store_true",
        help="whether to mask eos token loss or not; prefer masking if training for DA",
    )
    parser.add_argument("--code_desired", nargs='+',type=str, required=True, help='the desired label for generation')
    parser.add_argument("--multi_eval",  action="store_true", help='do multi aspect evaluation')
    parser.add_argument('--gen_length', type=int, default=20, help='the length of the generation.')
    parser.add_argument('--gen_do_sample', action="store_true", help='if do sampling for generation.')
    parser.add_argument('--gen_k', type=int, default=50, help='top-k filtering.')
    parser.add_argument('--gen_p', type=float, default=1.0, help='top-p filtering.')
    parser.add_argument('--gen_temperature', type=float, default=1.0, help='temperature for generation.')
    parser.add_argument('--gen_repetition_penalty', type=float, default=1.0, help='repition penalty for generation.')
    parser.add_argument('--gen_num_return_sequences', type=int, default=10, help='num of return sequences for generation.')
    # parser.add_argument("--add_sep", action="store_true",
    #                     help="Include sep token if this arg is used between the two sentences in a pair | can/should be used for mrpc/mnli/qqp/qnli")
    # parser.add_argument("--sst5", action="store_true",
    #                     help="custom ops for SST-5")
    # parser.add_argument("--jigsaw", action="store_true", help="custom setup for jigsaw")
    # parser.add_argument("--jigsaw_no_toxic_gen", action="store_true", help="custom setup for jigsaw - gen_loss used only for non-toxic samples | check training loop")
    # parser.add_argument("--code_0", type=str, default="negative", help="control code to be used for code 1 of 2 (we support 3 at most - with the third one = 'neutral' for now)")
    # parser.add_argument("--code_1", type=str, default="positive", help="control code to be used for code 2 of 2 (we support 3 at most - with the third one = 'neutral' for now)")

#     args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    #
    # # Setup distant debugging if needed
    # if args.server_ip and args.server_port:
    #     # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    #     import ptvsd
    #
    #     print("Waiting for debugger attach")
    #     ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    #     ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device



    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        # args.fp16,
    )

    # Set seed
    set_seed(args)


    # Prepare GLUE task
    #args.task_name = args.task_name.lower()
    # if args.task_name not in processors:
    #     raise ValueError("Task not found: %s" % (args.task_name))
    # processor = processors[args.task_name]()
    # args.output_mode = output_modes[args.task_name]
    # label_list = processor.get_labels()
    # num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    if args.new_encoder:
        model_class=GPT2LMHeadSemiPrefixLinearGSPartialModel_newenc
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.outbias:
    #    if args.threeway:
    #        config.nbias=3
    #    else:
        config.nbias=2
    else:
        config.nbias=0

    config.embd_pdrop = args.dropout
    config.attn_pdrop = args.dropout
    config.resid_pdrop = args.dropout
    if args.logit_scale:
        config.logit_scale=True
    else:
        config.logit_scale=False
    config.prefix_length=args.prefix_length
    config.prefix_hidden_size=args.prefix_hidden_size
    args.prefix_num=sum(args.train_task_ncat_list)
    config.prefix_num=args.prefix_num
    config.train_task_ncat_list=args.train_task_ncat_list
    config.train_task_prefix_nfilter = args.train_task_prefix_nfilter
    config.train_task_prefix_pfilter = args.train_task_prefix_pfilter
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.model_type == 'gpt2': #setting pad token for GPT-2
        tokenizer.pad_token = '[PAD]'
    # if args.add_sep:
    #     special_tokens_dict = {'sep_token': '<SEP>'}
    #     tokenizer.add_special_tokens(special_tokens_dict)
    config.output_past = True #https://github.com/huggingface/transformers/pull/3734
    config.pad_token_id = tokenizer.pad_token_id
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model.resize_token_embeddings(len(tokenizer))

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab


    model.to(args.device)




    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.output_sentiment_dir is not None:
            assert(args.train_task_list[0]=='sentiment')
            with torch.no_grad():
                model_sentiment = GPT2LMHeadPrefixModel.from_pretrained(args.output_sentiment_dir)
                model.prefix_theta_prime[0, :] = model_sentiment.prefix_neg_theta_prime
                model.prefix_theta_prime[1, :] = model_sentiment.prefix_pos_theta_prime
                model.prefix_mlp[0].weight.copy_(model_sentiment.prefix_neg_mlp.weight)
                model.prefix_mlp[1].weight.copy_(model_sentiment.prefix_pos_mlp.weight)
                del model_sentiment
                model_topic = GPT2LMHeadSemiPrefixLinearGSSepModel.from_pretrained(args.output_topic_dir)
                for prefix_idx in range(args.train_task_ncat_list[-1]):
                    model.prefix_theta_prime[prefix_idx + 2, :] = model_topic.prefix_theta_prime[prefix_idx , :]
                    model.prefix_mlp[2 + prefix_idx].weight.copy_(model_topic.prefix_mlp[prefix_idx ].weight)
                del model_topic
                logger.info("sentiment prefix loaded from %s", args.output_sentiment_dir)
                logger.info("topic prefix loaded from %s", args.output_topic_dir)
        if args.output_encoder_dir is not None:
            with torch.no_grad():
                model_encoder = GPT2LMHeadSemiPrefixLinearGSPartialModel.from_pretrained(args.output_encoder_dir)
                model.linear_hidden.weight.copy_(model_encoder.linear_hidden.weight)
                del model_encoder
        train_dataset = load_and_cache_examples(args, args.train_file_path, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    # results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        #tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        if args.model_type == 'gpt2' and tokenizer.pad_token is None: #setting pad token for GPT-2
            tokenizer.pad_token = '[PAD]'
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            # prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            evaluate(args, model, tokenizer, global_step, output_dir=args.eval_output_dir)
            # result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            # results.update(result)

    # return results


if __name__ == "__main__":
    main()
