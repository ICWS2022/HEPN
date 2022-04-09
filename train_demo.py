from fewshot_re_kit.data_loader import get_loader
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import BERTSentenceEncoder
import models
from models.proto import Proto
import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train_wiki',
            help='train file')
    parser.add_argument('--val', default='val_wiki',
            help='val file')
    parser.add_argument('--test', default='val_wiki',
            help='test file')
    parser.add_argument('--trainN', default=5, type=int,
            help='N in train')
    parser.add_argument('--N', default=5, type=int,
            help='N way')
    parser.add_argument('--K', default=1, type=int,
            help='K shot')
    parser.add_argument('--Q', default=15, type=int,
            help='Num of query per class')
    parser.add_argument('--batch_size', default=1, type=int,
            help='batch size')
    parser.add_argument('--train_iter', default=20000, type=int,
            help='num of iters in training')
    parser.add_argument('--val_iter', default=1000, type=int,
            help='num of iters in validation')
    parser.add_argument('--test_iter', default=10000, type=int,
            help='num of iters in testing')
    parser.add_argument('--val_step', default=2000, type=int,
           help='val after training how many iters')
    parser.add_argument('--model', default='proto',
            help='model name')
    parser.add_argument('--encoder', default='bert',
            help='encoder: bert')
    parser.add_argument('--max_length', default=128, type=int,
           help='max length')
    parser.add_argument('--lr', default=-1, type=float,
           help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
           help='weight decay')
    parser.add_argument('--grad_iter', default=1, type=int,
           help='accumulate gradient every x iterations')
    parser.add_argument('--optim', default='adam',
           help='sgd / adam / adamw')
    parser.add_argument('--load_ckpt', default=None,
           help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
           help='save ckpt')
    parser.add_argument('--only_test', action='store_true',
           help='only test')
    parser.add_argument('--ckpt_name', type=str, default='',
           help='checkpoint name.')



    parser.add_argument('--pretrain_ckpt', default=None,
           help='bert pre-trained checkpoint')
    parser.add_argument('--cat_entity_rep', action='store_true',
           help='concatenate entity representation as sentence rep')
    parser.add_argument('--mask_entity', action='store_true',
           help='mask entity names')
    parser.add_argument('--use_sgd_for_bert', action='store_true',
           help='use SGD instead of AdamW for BERT.')

    opt = parser.parse_args()
    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    encoder_name = opt.encoder
    max_length = opt.max_length

    
    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))
    

    pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
    sentence_encoder = BERTSentenceEncoder(
            pretrain_ckpt,
            max_length,
            cat_entity_rep=opt.cat_entity_rep,
            mask_entity=opt.mask_entity)

    

    train_data_loader = get_loader(opt.train, sentence_encoder, N=trainN, K=K, Q=Q, batch_size=batch_size)
    val_data_loader = get_loader(opt.val, sentence_encoder, N=N, K=K, Q=Q, batch_size=batch_size)
    test_data_loader = get_loader(opt.test, sentence_encoder, N=N, K=K, Q=Q, batch_size=batch_size)
   
    if opt.optim == 'sgd':
        pytorch_optim = optim.SGD
    elif opt.optim == 'adam':
        pytorch_optim = optim.Adam
    elif opt.optim == 'adamw':
        from transformers import AdamW
        pytorch_optim = AdamW
    else:
        raise NotImplementedError

    framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)
        
    prefix = '-'.join([model_name, encoder_name, opt.train, opt.val, str(N), str(K)])
    if opt.cat_entity_rep:
        prefix += '-catentity'
    if len(opt.ckpt_name) > 0:
        prefix += '-' + opt.ckpt_name
    
    if model_name == 'proto':
        model = Proto(sentence_encoder)
    else:
        raise NotImplementedError
    
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if torch.cuda.is_available():
        model.cuda()

    if not opt.only_test:
        if encoder_name in ['bert']:
            bert_optim = True
        else:
            bert_optim = False

        if opt.lr == -1:
            if bert_optim:
                opt.lr = 2e-5
            else:
                opt.lr = 1e-1
        
        opt.train_iter = opt.train_iter * opt.grad_iter
        framework.train(model, prefix, batch_size, trainN, N, K, Q,
                pytorch_optim=pytorch_optim, load_ckpt=opt.load_ckpt, save_ckpt=ckpt, val_step=opt.val_step,
                train_iter=opt.train_iter, val_iter=opt.val_iter, bert_optim=bert_optim, 
                learning_rate=opt.lr, use_sgd_for_bert=opt.use_sgd_for_bert, grad_iter=opt.grad_iter)
    else:
        ckpt = opt.load_ckpt
        if ckpt is None:
            print("Warning: --load_ckpt is not specified. Will load Hugginface pre-trained checkpoint.")
            ckpt = 'none'

    acc = framework.eval(model, batch_size, N, K, Q, opt.test_iter, ckpt=ckpt)
    print("RESULT: %.2f" % (acc * 100))

if __name__ == "__main__":
    main()
