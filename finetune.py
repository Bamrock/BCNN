import glob
import json
import logging
import os
import re
import shutil
import random
from multiprocessing import Pool
from typing import Dict, List, Tuple
from copy import deepcopy
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import argparse

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertForLongSequenceClassification,
    BertForLongSequenceClassificationCat,
    BertTokenizer,
    DNATokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    FlaubertConfig,
    FlaubertForSequenceClassification,
    FlaubertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
# from transformers import glue_convert_pred_examples_to_features as convert_pred_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
        BertConfig,
        XLNetConfig,
        XLMConfig,
        RobertaConfig,
        DistilBertConfig,
        AlbertConfig,
        XLMRobertaConfig,
        FlaubertConfig,
    )
    ),
    (),
)


MODEL_CLASSES = {
    "dna": (BertConfig, BertForSequenceClassification, DNATokenizer),
    "dnalong": (BertConfig, BertForLongSequenceClassification, DNATokenizer),
    "dnalongcat": (BertConfig, BertForLongSequenceClassificationCat, DNATokenizer),
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
}

TOKEN_ID_GROUP = ["bert", "dnalong", "dnalongcat", "xlnet", "albert"]

class Config(object):
    def __init__(self):
        self.data_dir = './data/'
        self.model_type = 'dna'
        self.n_process = 20
        self.should_continue = False
        self.model_name_or_path = ''
        self.task_name = 'dnaprom'
        self.output_dir = ''
        self.visualize_data_dir = None
        self.result_dir = './result'
        self.config_name = './transformers/dnabert-config/bert-config-5/config.json'
        self.tokenizer_name = 'dna5'
        self.cache_dir = ''
        self.predict_dir = './predict'
        self.max_seq_length = 200
        self.do_train = True
        self.do_eval = False
        self.do_predict = False
        self.do_visualize = False
        self.visualize_train = False
        self.do_ensemble_pred = False
        self.evaluate_during_training = True
        self.do_lower_case = False
        self.per_gpu_train_batch_size = 25
        self.per_gpu_eval_batch_size = 25
        self.per_gpu_pred_batch_size = 25
        self.early_stop = 0
        self.predict_scan_size = 1
        self.gradient_accumulation_steps = 20
        self.learning_rate = 5e-4
        # self.weight = [100, 10, 1]
        self.weight = None
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.max_grad_norm = 1.0
        self.attention_probs_dropout_prob = 0.1
        self.hidden_dropout_prob = 0.1
        self.rnn_dropout = 0.0
        self.rnn = 'lstm'
        self.num_rnn_layer = 2
        self.rnn_hidden = 768
        self.num_train_epochs = 20.0
        self.max_steps = -1
        self.warmup_steps = 50
        self.warmup_percent = 0.0
        self.logging_steps = 1000
        self.save_steps = -1
        self.save_total_limit = None
        self.eval_all_checkpoints = True
        self.no_cuda = False
        self.overwrite_output_dir = True
        self.overwrite_cache = True
        self.visualize_models = None
        self.seed = random.randint(1,10000)
        self.fp16 = False
        self.fp16_opt_level = "O1"
        self.local_rank = -1
        self.server_ip = ''
        self.server_port = ''
        self.n_gpu = 1

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        # logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    # args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    warmup_steps = args.warmup_steps if args.warmup_percent == 0 else int(args.warmup_percent * t_total)

    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,
    #                   betas=(args.beta1, args.beta2))

    ignored_params = list(map(id, model.classifier.parameters()))  # 返回的是parameters的 内存地址
    base_params = list(filter(lambda p: id(p) not in ignored_params, model.parameters()))

    optimizer = AdamW([
        {'params': base_params},
        {'params': model.classifier.parameters(), 'lr': args.learning_rate}], lr=args.learning_rate / 10, eps=args.adam_epsilon,
                      betas=(args.beta1, args.beta2))

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    # logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    # logger.info(
    #     "  Total train batch size (w. parallel, distributed & accumulation) = %d",
    #     args.train_batch_size
    #     * args.gradient_accumulation_steps
    #     * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    # )
    # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    # logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # # Check if continuing training from a checkpoint
    # if os.path.exists(args.model_name_or_path):
    #     # set global_step to gobal_step of last saved checkpoint from model path
    #     global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
    #     print(global_step, len(train_dataloader), args.gradient_accumulation_steps)
    #     epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
    #     steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        # logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        # logger.info("  Continuing training from epoch %d", epochs_trained)
        # logger.info("  Continuing training from global step %d", global_step)
        # logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    # train_iterator = trange(
    #     epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    # )
    train_iterator = int(args.num_train_epochs)
    set_seed(args)  # Added here for reproductibility

    ####variable for saving best specificity
    best_auc = 0
    best_spec = 0
    last_spec = 0
    stop_count = 0
    best_spec_acc = 0
    best_spec_sens = 0
    best_spec_f1 = 0
    acc_HD = 0
    acc_healthy = 0
    ####variable for saving best sensitity
    best_sens = 0
    best_sens_spec = 0
    best_sens_acc = 0
    best_sens_f1 = 0
    ####variable for saving best acc
    best_acc = 0
    best_acc_sens = 0
    best_acc_spec = 0
    best_acc_f1 = 0
    ####variable for saving best f1
    best_f1 = 0
    best_f1_sens = 0
    best_f1_spec = 0
    best_f1_acc = 0



    for _ in range(train_iterator):
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        epoch_iterator = train_dataloader
        for step, batch in enumerate(epoch_iterator):

            # # Skip past any already trained steps if resuming training
            # if steps_trained_in_current_epoch > 0:
            #     steps_trained_in_current_epoch -= 1
            #     continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in TOKEN_ID_GROUP else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

            if step%1000 == 0 and step <= len(train_dataloader)//1000:
                logs = {}
                results = evaluate(args, model, tokenizer)
                # for key, value in results.items():
                #     eval_key = "eval_{}".format(key)
                #     logs[eval_key] = value
                #
                # if args.task_name == "dna690" and results["auc"] < best_auc:
                #     continue

                ### save specificity >= 99.99%
                # if results['spec'] >= 0.9999 and results['sens'] >= best_spec_sens:
                if results['spec'] > best_spec and results['sens'] >=0.7:
                    best_spec = results['spec']
                    best_spec_acc = results['acc']
                    best_spec_sens = results['sens']
                    # acc_HD = results['acc_HD']
                    # acc_healthy = results['acc_healthy']
                    # best_spec_f1 = results['f1']
                    # output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
                    output_dir_dmr = os.path.join(args.output_dir, args.dmr, "best_specificity")
                    if not os.path.exists(output_dir_dmr):
                        os.makedirs(output_dir_dmr)
                    output_eval_file = os.path.join(output_dir_dmr, "eval_results.txt")
                    with open(output_eval_file, "a") as writer:
                        if args.task_name[:3] == "dna":
                            eval_result = args.data_dir.split('/')[-1] + " "
                        else:
                            eval_result = ""
                        logger.info("***** Eval results *****")
                        # for key in sorted(result.keys()):
                        #     logger.info("  %s = %s", key, str(result[key]))
                        #     eval_result = eval_result + str(result[key])[:5] + " "
                        # writer.write(n + '\t' + eval_result + "\n")
                        for key in ['acc', 'spec', 'sens']:
                            logger.info("  %s = %s", key, str(results[key]))
                            eval_result = eval_result + str(results[key])[:5] + "\t"
                        writer.write(args.dmr + '\t' + eval_result + "\n")
                    # checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    # output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    output_dir = os.path.join(args.output_dir, "{}".format(args.dmr), "best_specificity")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    # logger.info("Saving model checkpoint to %s", output_dir)
                    # _rotate_checkpoints(args, checkpoint_prefix)
                    if args.task_name != "dna690":
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    # logger.info("Saving optimizer and scheduler states to %s", output_dir)
                elif results['spec'] == best_spec and results['sens'] >= best_spec_sens:
                    best_spec = results['spec']
                    best_spec_acc = results['acc']
                    best_spec_sens = results['sens']
                    # acc_HD = results['acc_HD']
                    # acc_healthy = results['acc_healthy']
                    # best_spec_f1 = results['f1']
                    # output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
                    output_dir_dmr = os.path.join(args.output_dir, args.dmr, "best_specificity")
                    if not os.path.exists(output_dir_dmr):
                        os.makedirs(output_dir_dmr)
                    output_eval_file = os.path.join(output_dir_dmr, "eval_results.txt")
                    with open(output_eval_file, "a") as writer:
                        if args.task_name[:3] == "dna":
                            eval_result = args.data_dir.split('/')[-1] + " "
                        else:
                            eval_result = ""
                        logger.info("***** Eval results *****")
                        # for key in sorted(result.keys()):
                        #     logger.info("  %s = %s", key, str(result[key]))
                        #     eval_result = eval_result + str(result[key])[:5] + " "
                        # writer.write(n + '\t' + eval_result + "\n")
                        for key in ['acc', 'spec', 'sens']:
                            logger.info("  %s = %s", key, str(results[key]))
                            eval_result = eval_result + str(results[key])[:5] + "\t"
                        writer.write(args.dmr + '\t' + eval_result + "\n")
                    # checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    # output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    output_dir = os.path.join(args.output_dir, "{}".format(args.dmr), "best_specificity")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    # logger.info("Saving model checkpoint to %s", output_dir)
                    # _rotate_checkpoints(args, checkpoint_prefix)
                    if args.task_name != "dna690":
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    # logger.info("Saving optimizer and scheduler states to %s", output_dir)


               

                logs["best_spec"] = best_spec
                logs["best_spec_acc"] = best_spec_acc
                logs["best_spec_sens"] = best_spec_sens
               


                loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                learning_rate_scalar = scheduler.get_lr()[0]
                logs["learning_rate"] = learning_rate_scalar
                logs["loss"] = loss_scalar
                logging_loss = tr_loss

                for key, value in logs.items():
                    tb_writer.add_scalar(key, value, global_step)
                print(json.dumps({**logs, **{"step": global_step}}))


        if args.early_stop != 0:
            # record current auc to perform early stop
            if results["acc"] == last_acc:
                stop_count += 1
            else:
                stop_count = 0
            last_acc = results["acc"]
            if stop_count == args.early_stop:
                logger.info("Early stop")
                train_iterator.close()
                break
                # return global_step, tr_loss / global_step
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", evaluate=True):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)
    if args.task_name[:3] == "dna":
        softmax = torch.nn.Softmax(dim=1)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=evaluate)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        # args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        args.eval_batch_size = args.per_gpu_eval_batch_size
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        probs = None
        out_label_ids = None
        # for batch in tqdm(eval_dataloader, desc="Evaluating"):
        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in TOKEN_ID_GROUP else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            if args.task_name[:3] == "dna" and args.task_name != "dnasplice":
                if args.do_ensemble_pred:
                    probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()
                else:
                    probs = softmax(torch.tensor(preds, dtype=torch.float32))[:, 1].numpy()
            elif args.task_name == "dnasplice":
                probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        if args.do_ensemble_pred:
            result = compute_metrics(eval_task, preds, out_label_ids, probs[:, 1])
        else:
            result = compute_metrics(eval_task, preds, out_label_ids, probs)
        results.update(result)

        if args.task_name == "dna690":
            eval_output_dir = args.result_dir
            if not os.path.exists(args.result_dir):
                os.makedirs(args.result_dir)
     


    if args.do_ensemble_pred:
        return results, eval_task, preds, out_label_ids, probs
    else:
        return results


def predict(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)
    softmax = torch.nn.Softmax(dim=1)

    predictions = {}
    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=True)

        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        pred_loss = 0.0
        nb_pred_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(pred_dataloader, desc="Predicting"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in TOKEN_ID_GROUP else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                _, logits = outputs[:2]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        if args.output_mode == "classification":
            if args.task_name[:3] == "dna" and args.task_name != "dnasplice":
                if args.do_ensemble_pred:
                    probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()
                else:
                    probs = softmax(torch.tensor(preds, dtype=torch.float32))[:, 1].numpy()
            elif args.task_name == "dnasplice":
                probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)

        if args.do_ensemble_pred:
            result = compute_metrics(pred_task, preds, out_label_ids, probs[:, 1])
        else:
            result = compute_metrics(pred_task, preds, out_label_ids, probs)

        df1 = pd.DataFrame(probs)
        df1.columns = ['prediction_tumor']
        df1['prediction_normal'] = 1 - df1['prediction_tumor']
        # df1.columns = ['prediction_normal']
        df2 = pd.DataFrame(out_label_ids)
        df2.columns = ['label']
        df3 = pd.DataFrame(preds)
        df3.columns = ['pred_label']
        df_final = pd.concat([df1, df2, df3], axis=1)
        df_final['DMR'] = args.dmr
        df_final = df_final[['DMR', 'prediction_normal', 'prediction_tumor', 'label', 'pred_label']]
        df_final.to_csv(args.csv, index=False)

        pred_output_dir = args.predict_dir
        if not os.path.exists(pred_output_dir):
            os.makedir(pred_output_dir)
        output_pred_file = os.path.join(pred_output_dir, "pred_results.npy")
        logger.info("***** Pred results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        np.save(output_pred_file, probs)


def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed).unsqueeze(0)


def visualize(args, model, tokenizer, kmer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.predict_dir,)
    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)
    softmax = torch.nn.Softmax(dim=1)

    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        '''
        if args.task_name != "dna690":
            args.data_dir = os.path.join(args.visualize_data_dir, str(kmer))
        else:
            args.data_dir = deepcopy(args.visualize_data_dir).replace("/690", "/690/" + str(kmer))
        '''

        evaluate = False if args.visualize_train else True
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, evaluate=evaluate)

        if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(pred_output_dir)

        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        pred_loss = 0.0
        nb_pred_steps = 0
        batch_size = args.pred_batch_size
        if args.task_name != "dnasplice":
            preds = np.zeros([len(pred_dataset), 2])
        else:
            preds = np.zeros([len(pred_dataset), 3])
        attention_scores = np.zeros([len(pred_dataset), 12, args.max_seq_length, args.max_seq_length])

        for index, batch in enumerate(tqdm(pred_dataloader, desc="Predicting")):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in TOKEN_ID_GROUP else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                attention = outputs[-1][-1]
                _, logits = outputs[:2]

                preds[index * batch_size:index * batch_size + len(batch[0]), :] = logits.detach().cpu().numpy()
                attention_scores[index * batch_size:index * batch_size + len(batch[0]), :, :,
                :] = attention.cpu().numpy()
                # if preds is None:
                #     preds = logits.detach().cpu().numpy()
                # else:
                #     preds = np.concatenate((preds, logits.detach().cpu().numpy()), axis=0)

                # if attention_scores is not None:
                #     attention_scores = np.concatenate((attention_scores, attention.cpu().numpy()), 0)
                # else:
                #     attention_scores = attention.cpu().numpy()

        if args.task_name != "dnasplice":
            probs = softmax(torch.tensor(preds, dtype=torch.float32))[:, 1].numpy()
        else:
            probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()

        scores = np.zeros([attention_scores.shape[0], attention_scores.shape[-1]])

        for index, attention_score in enumerate(attention_scores):
            attn_score = []
            for i in range(1, attention_score.shape[-1] - kmer + 2):
                attn_score.append(float(attention_score[:, 0, i].sum()))

            for i in range(len(attn_score) - 1):
                if attn_score[i + 1] == 0:
                    attn_score[i] = 0
                    break

            # attn_score[0] = 0
            counts = np.zeros([len(attn_score) + kmer - 1])
            real_scores = np.zeros([len(attn_score) + kmer - 1])
            for i, score in enumerate(attn_score):
                for j in range(kmer):
                    counts[i + j] += 1.0
                    real_scores[i + j] += score
            real_scores = real_scores / counts
            real_scores = real_scores / np.linalg.norm(real_scores)

            # print(index)
            # print(real_scores)
            # print(len(real_scores))

            scores[index] = real_scores

    return scores, probs


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if args.do_predict:
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}".format(
                "dev" if evaluate else "train",
                str(args.max_seq_length),
                str(task),
            ),
        )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        # logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # logger.info("Creating features from dataset file at %s", args.data_dir)
        # label_list = processor.get_labels()
        label_list = processor.get_two_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        )
        # print("***************************", examples)
        print("finish loading examples")

        # params for convert_examples_to_features
        max_length = args.max_seq_length
        pad_on_left = bool(args.model_type in ["xlnet"])
        pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        pad_token_segment_id = 4 if args.model_type in ["xlnet"] else 0

        if args.n_process == 1:
            features = convert_examples_to_features(
                examples,
                tokenizer,
                label_list=label_list,
                max_length=max_length,
                output_mode=output_mode,
                pad_on_left=pad_on_left,  # pad on the left for xlnet
                pad_token=pad_token,
                pad_token_segment_id=pad_token_segment_id, )

        else:
            n_proc = int(args.n_process)
            if evaluate:
                n_proc = max(int(n_proc / 4), 1)
            print("number of processes for converting feature: " + str(n_proc))
            p = Pool(n_proc)
            indexes = [0]
            len_slice = int(len(examples) / n_proc)
            for i in range(1, n_proc + 1):
                if i != n_proc:
                    indexes.append(len_slice * (i))
                else:
                    indexes.append(len(examples))

            results = []

            for i in range(n_proc):
                results.append(p.apply_async(convert_examples_to_features, args=(
                examples[indexes[i]:indexes[i + 1]], tokenizer, max_length, None, label_list, output_mode, pad_on_left,
                pad_token, pad_token_segment_id, True,)))
                print(str(i + 1) + ' processor started !')

            p.close()
            p.join()

            features = []
            for result in results:
                features.extend(result.get())

        if args.local_rank in [-1, 0]:
            # logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset



def main(n):
    args = Config()
    args.dmr = n
    args.data_dir = ''
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

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

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        # args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device



    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    # label_list = processor.get_labels()
    label_list = processor.get_two_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if not args.do_visualize and not args.do_ensemble_pred:
        config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
        if args.model_type in ["dnalong", "dnalongcat"]:
            assert args.max_seq_length % 512 == 0
        config.split = int(args.max_seq_length / 512)
        config.rnn = args.rnn
        config.num_rnn_layer = args.num_rnn_layer
        config.rnn_dropout = args.rnn_dropout
        config.rnn_hidden = args.rnn_hidden
        config.weight = args.weight

        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        # logger.info('finish loading model')

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model.to(args.device)

        # logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

   

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(os.path.join(args.output_dir, "{}".format(args.dmr)), do_lower_case=args.do_lower_case)
        # tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            # logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        # logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    # Prediction
    predictions = {}
    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoint = args.output_dir
        # logger.info("Predict using the following checkpoint: %s", checkpoint)
        prefix = ''
        model = model_class.from_pretrained(checkpoint)
        model.to(args.device)
        prediction = predict(args, model, tokenizer, prefix=prefix)

    # Visualize
    if args.do_visualize and args.local_rank in [-1, 0]:
        visualization_models = [3, 4, 5, 6] if not args.visualize_models else [args.visualize_models]

        scores = None
        all_probs = None

        for kmer in visualization_models:
            output_dir = args.output_dir.replace("/690", "/690/" + str(kmer))
            # checkpoint_name = os.listdir(output_dir)[0]
            # output_dir = os.path.join(output_dir, checkpoint_name)

            tokenizer = tokenizer_class.from_pretrained(
                "dna" + str(kmer),
                do_lower_case=args.do_lower_case,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            checkpoint = output_dir
            # logger.info("Calculate attention score using the following checkpoint: %s", checkpoint)
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            config = config_class.from_pretrained(
                output_dir,
                num_labels=num_labels,
                finetuning_task=args.task_name,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            config.output_attentions = True
            model = model_class.from_pretrained(
                checkpoint,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            model.to(args.device)
            attention_scores, probs = visualize(args, model, tokenizer, prefix=prefix, kmer=kmer)
            if scores is not None:
                all_probs += probs
                scores += attention_scores
            else:
                all_probs = deepcopy(probs)
                scores = deepcopy(attention_scores)

        all_probs = all_probs / float(len(visualization_models))
        np.save(os.path.join(args.predict_dir, "atten.npy"), scores)
        np.save(os.path.join(args.predict_dir, "pred_results.npy"), all_probs)

    # ensemble prediction
    if args.do_ensemble_pred and args.local_rank in [-1, 0]:

        for kmer in range(3, 7):
            output_dir = os.path.join(args.output_dir, str(kmer))
            tokenizer = tokenizer_class.from_pretrained(
                "dna" + str(kmer),
                do_lower_case=args.do_lower_case,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            checkpoint = output_dir
            # logger.info("Calculate attention score using the following checkpoint: %s", checkpoint)
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            config = config_class.from_pretrained(
                output_dir,
                num_labels=num_labels,
                finetuning_task=args.task_name,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            config.output_attentions = True
            model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            model.to(args.device)
            if kmer == 3:
                args.data_dir = os.path.join(args.data_dir, str(kmer))
            else:
                args.data_dir = args.data_dir.replace("/" + str(kmer - 1), "/" + str(kmer))

            if args.result_dir.split('/')[-1] == "test.npy":
                results, eval_task, _, out_label_ids, probs = evaluate(args, model, tokenizer, prefix=prefix)
            elif args.result_dir.split('/')[-1] == "train.npy":
                results, eval_task, _, out_label_ids, probs = evaluate(args, model, tokenizer, prefix=prefix,
                                                                       evaluate=False)
            else:
                raise ValueError("file name in result_dir should be either test.npy or train.npy")

            if kmer == 3:
                all_probs = deepcopy(probs)
                cat_probs = deepcopy(probs)
            else:
                all_probs += probs
                cat_probs = np.concatenate((cat_probs, probs), axis=1)
            print(cat_probs[0])

        all_probs = all_probs / 4.0
        all_preds = np.argmax(all_probs, axis=1)

        # save label and data for stuck ensemble
        labels = np.array(out_label_ids)
        labels = labels.reshape(labels.shape[0], 1)
        data = np.concatenate((cat_probs, labels), axis=1)
        random.shuffle(data)
        root_path = args.result_dir.replace(args.result_dir.split('/')[-1], '')
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        np.save(args.result_dir, data)
        ensemble_results = compute_metrics(eval_task, all_preds, out_label_ids, all_probs[:, 1])
        logger.info("***** Ensemble results {} *****".format(prefix))
        for key in sorted(ensemble_results.keys()):
            logger.info("  %s = %s", key, str(ensemble_results[key]))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--dmr', type=str, required=True, help='')
    # parser.add_argument('-c', '--cancer', type=int, required=True, help='')
    # parser.add_argument('-n', '--normal', type=int, required=True, help='')
    # parser.add_argument('-o', '--output', type=str, required=True, help='')
    args = parser.parse_args()
    # weight = [args.cancer, args.normal]
    n = args.dmr
    # output_dir = args.output
    # main(weight, output_dir)
    main(n)
