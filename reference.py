import torch
from models.r2gen_en import R2GenModel as R2GenModelen
from models.r2gen_ch import R2GenModel as R2GenModelch
from PIL import Image
from modules.tokenizers import Tokenizer
#import main
import argparse
import json
import re
from collections import Counter
import numpy as np
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
import torch.nn as nn

def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/cov/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/cov/annotation.json', help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='cov', choices=['iu_xray', 'mimic_cxr','cov'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=100, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=2, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=6, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
    # for Relational Memory
    parser.add_argument('--rm_num_slots', type=int, default=3, help='the number of memory slots.')
    parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
    parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/cov', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')

    args = parser.parse_args()
    return args



args = parse_agrs()
tokenizer = Tokenizer(args)

# create data loader
train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

# build model architecture
model_en = R2GenModelen(args, tokenizer).to('cuda' if torch.cuda.is_available() else 'cpu')
model_ch = R2GenModelch(args, tokenizer).to('cuda' if torch.cuda.is_available() else 'cpu')
# get function handles of loss and metrics
criterion = nn.CrossEntropyLoss(ignore_index=0)
metric_ftns = compute_scores

current=1
device=('cuda' if torch.cuda.is_available() else 'cpu')
if current:
    state_dict = torch.load('checkpoint/current_checkpoint.pth')
    model_state_dict = state_dict['state_dict']
    model_en.load_state_dict(model_state_dict, strict=False)
    model_ch.load_state_dict(model_state_dict, strict=False)
    model_en.to(torch.device('cuda'))
    model_ch.to(torch.device('cuda'))
else:
    state_dict = torch.load('checkpoint/model_best.pth')
    model_state_dict = state_dict['state_dict']
    model_en.load_state_dict(model_state_dict, strict=False)
    model_ch.load_state_dict(model_state_dict, strict=False)
    model_en.to(torch.device('cuda'))
    model_ch.to(torch.device('cuda'))


def greedy_decoder(model, report, reports_ids, start_symbol):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    # enc_outputs, enc_self_attns = model.encoder(enc_outputs)
    dec_input = torch.zeros(1, 0).type_as(reports_ids.data)
    #print(345,report)
    #print(223, dec_input)
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        dec_input = torch.cat([dec_input.detach(), torch.tensor([[next_symbol]], dtype=reports_ids.dtype).cuda()], -1)
        #dec_input = torch.tensor([[next_symbol]], dtype=reports_ids.dtype)
        #print('aaa',dec_input)

        dec_outputs = model(report, dec_input)
        #print('22223',dec_outputs.size())
        #dec_outputs = model(report, 1)

        #projected = model.projection(dec_outputs)
        projected=dec_outputs  # torch.Size([0, 402])
        projected=projected.unsqueeze(0)
        #print('projected',projected.size())
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]

        #print('prob',prob.size())
        next_word = prob.data[-1]

        next_symbol = next_word
        if next_symbol==3 or dec_input.size(1) >=99:
            terminal = True
        #print(next_word)
    return dec_input



#验证集
#model.eval()
model_en.eval()
model_ch.eval()
with torch.no_grad():
    result_report_val_english = []
    result_report_val_chinese = []
    val_gts_english, val_gts_chinese = [], []
    val_res_english, val_res_chinese = [], []
    ground_truths_eng, ground_truths_cn = [], []

    for batch_idx, (images_id, images, reports_ids, report, image_path_all, reports_ids_use) in enumerate(
            val_dataloader):
        images, reports_ids, reports_ids_use = images.to(device), reports_ids.to(
            device), reports_ids_use.to(device)

        for i in range(len(images_id)):
            # print(456,reports_ids[i][0])
            if reports_ids[i][0]==1:
                greedy_dec_input = greedy_decoder(model_en, image_path_all[i],  reports_ids[i], start_symbol=1)
                predict = model_en(image_path_all[i], greedy_dec_input)
                predict = predict.data.max(1, keepdim=True)[1]

                # predict = predict.data.max(1, keepdim=True)[1]
                predict = predict.squeeze()
                reports = model_en.tokenizer.decode(predict.cpu().numpy())
                temp = {'reports_ids': images_id[i], 'reports': reports, 'label': 'english'}
                result_report_val_english.append(temp)
                val_res_english.append(reports)
                gtt = model_en.tokenizer.decode(reports_ids[i][1:].cpu().numpy())
                ground_truths_eng.append(gtt)

                val_gts_english = ground_truths_eng

    val_met_english = metric_ftns({i: [gt] for i, gt in enumerate(val_gts_english)},
                                  {i: [re] for i, re in enumerate(val_res_english)})

    for key, value in val_met_english.items():
        print('val_english' +key,value)
    resFiletest = 'valreports/english-' + '.json'
    json.dump(result_report_val_english, open(resFiletest, 'w'))





    for batch_idx, (images_id, images, reports_ids, report, image_path_all, reports_ids_use) in enumerate(
            val_dataloader):
        images, reports_ids, reports_ids_use = images.to(device), reports_ids.to(
            device), reports_ids_use.to(device)

        for i in range(len(images_id)):
            if reports_ids[i][0] == 2:
                greedy_dec_input = greedy_decoder(model_ch, image_path_all[i], reports_ids[i], start_symbol=2)
                predict = model_ch(image_path_all[i], greedy_dec_input)
                predict = predict.data.max(1, keepdim=True)[1]

                predict = predict.squeeze()
                reports = model_ch.tokenizer.decode(predict.cpu().numpy())
                reports = reports.replace(" ", "")
                temp = {'reports_ids': images_id[i], 'reports': reports, 'label': 'chinese'}
                result_report_val_chinese.append(temp)
                reports = ' '.join(reports)
                val_res_chinese.append(reports)
                gtt1 = model_ch.tokenizer.decode(reports_ids[i][1:].cpu().numpy())
                gtt1 = gtt1.replace(" ", "")
                gtt1 = ' '.join(gtt1)
                ground_truths_cn.append(gtt1)

                val_gts_chinese = ground_truths_cn

    val_met_chinese = metric_ftns({i: [gt] for i, gt in enumerate(val_gts_chinese)},
                                       {i: [re] for i, re in enumerate(val_res_chinese)})
    #log.update(**{'val_chinese' + k: v for k, v in val_met_chinese.items()})
    for key, value in val_met_chinese.items():
        print('val_chinese' + key, value)
    resFiletest = 'valreports/chinese-' + '.json'
    json.dump(result_report_val_chinese, open(resFiletest, 'w', encoding='utf-8'), ensure_ascii=False)




#测试集
#model.eval()
model_en.eval()
model_ch.eval()
with torch.no_grad():
    result_report_test_english = []
    result_report_test_chinese = []
    test_gts_english, test_gts_chinese, test_res_english, test_res_chinese = [], [], [], []
    ground_truths_eng, ground_truths_cn = [], []

    for batch_idx, (images_id, images, reports_ids, report, image_path_all, reports_ids_use) in enumerate(
            test_dataloader):
        images, reports_ids, reports_ids_use = images.to(device), reports_ids.to(
            device), reports_ids_use.to(device)

        for i in range(len(images_id)):
            if reports_ids[i][0] == 1:
                greedy_dec_input = greedy_decoder(model_en, image_path_all[i], reports_ids[i], start_symbol=1)
                predict = model_en(image_path_all[i], greedy_dec_input)
                predict = predict.data.max(1, keepdim=True)[1]

                predict = predict.squeeze()
                reports = model_en.tokenizer.decode(predict.cpu().numpy())
                temp = {'reports_ids': images_id[i], 'reports': reports, 'label': 'english'}
                result_report_test_english.append(temp)
                test_res_english.append(reports)
                gtt2 = model_en.tokenizer.decode(reports_ids[i][1:].cpu().numpy())
                ground_truths_eng.append(gtt2)

                test_gts_english = ground_truths_eng

    test_met_english = metric_ftns({i: [gt] for i, gt in enumerate(test_gts_english)},
                                   {i: [re] for i, re in enumerate(test_res_english)})

    for key, value in test_met_english.items():
        print('test_english' + key, value)

    resFiletest_eng = 'testreports/english-' + '.json'
    json.dump(result_report_test_english, open(resFiletest_eng, 'w'))


    for batch_idx, (images_id, images, reports_ids, report, image_path_all, reports_ids_use) in enumerate(
            test_dataloader):
        images, reports_ids, reports_ids_use = images.to(device), reports_ids.to(
            device), reports_ids_use.to(device)

        for i in range(len(images_id)):
            if reports_ids[i][0] == 2:
                greedy_dec_input = greedy_decoder(model_ch, image_path_all[i], reports_ids[i], start_symbol=2)
                predict = model_ch(image_path_all[i], greedy_dec_input)
                predict = predict.data.max(1, keepdim=True)[1]

                predict = predict.squeeze()
                reports = model_ch.tokenizer.decode(predict.cpu().numpy())
                reports = reports.replace(" ", "")
                temp = {'reports_ids': images_id[i], 'reports': reports, 'label': 'chinese'}
                result_report_test_chinese.append(temp)
                reports = ' '.join(reports)
                test_res_chinese.append(reports)
                gtt3 = model_ch.tokenizer.decode(reports_ids[i][1:].cpu().numpy())
                gtt3 = gtt3.replace(" ", "")
                gtt3 = ' '.join(gtt3)
                ground_truths_cn.append(gtt3)

                test_gts_chinese = ground_truths_cn

    test_met_chinese = metric_ftns({i: [gt] for i, gt in enumerate(test_gts_chinese)},
                                        {i: [re] for i, re in enumerate(test_res_chinese)})

    for key, value in test_met_chinese.items():
        print('test_chinese' + key, value)

    resFiletest = 'testreports/chinese-' + '.json'
    json.dump(result_report_test_chinese, open(resFiletest, 'w', encoding='utf-8'), ensure_ascii=False)

