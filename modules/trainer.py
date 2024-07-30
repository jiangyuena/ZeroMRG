import os
from abc import abstractmethod
import json
import time
import torch
import pandas as pd
from numpy import inf


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_english' + args.monitor_metric
        self.mnt_metric_test = 'test_english' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if epoch>=30:
            pot='current_checkpoint'+str(epoch)+'.pth'
            filename1 = os.path.join(self.checkpoint_dir, pot)
            torch.save(state, filename1)
            print("Saving checkpoint: {} ...".format(filename1))

        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))

if not os.path.exists('valreports/'):
    os.makedirs('valreports/')
if not os.path.exists('testreports/'):
    os.makedirs('testreports/')

class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader


    def greedy_decoder(self, model, report, reports_ids, start_symbol):
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



    def _train_epoch(self, epoch):

        train_loss = 0
        self.model.cuda().train()
        for batch_idx, (images_id, images, reports_ids, report,image_path_all,reports_ids_use) in enumerate(self.train_dataloader):
            images, reports_ids,reports_ids_use= images.to(self.device), reports_ids.to(self.device),reports_ids_use.to(self.device)

            output = self.model(report, reports_ids_use)
            
            #print(7777,reports_ids[:, 1:].size())
            #print(888,output[:,:-1].size())
            #dec_logits=dec_logits[:,:-1,:]
            
            #loss = self.criterion(output[:,:-1,:], reports_ids[:, 1:].reshape(-1))
            loss = self.criterion(output, reports_ids[:, 1:].reshape(-1))
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
        log = {'train_loss': train_loss / len(self.train_dataloader)}


        self.model.eval()
        with torch.no_grad():
            result_report_val_english = []
            result_report_val_chinese = []
            val_gts_english,val_gts_chinese=[], []
            val_res_english, val_res_chinese =  [] ,[]
            ground_truths_eng, ground_truths_cn=[], []

            for batch_idx, (images_id, images, reports_ids, report,image_path_all,reports_ids_use) in enumerate(self.val_dataloader):
                images, reports_ids,reports_ids_use= images.to(self.device), reports_ids.to(
                    self.device),reports_ids_use.to(self.device)

                for i in range(len(images_id)):
                    #print(456,reports_ids[i][0])
                    if  reports_ids[i][0]==1:
                        greedy_dec_input = self.greedy_decoder(self.model, report[i],reports_ids_use[i],start_symbol=1)
                        predict = self.model(report[i], greedy_dec_input)
                        predict=predict.data.max(1, keepdim=True)[1]

                   	    #predict = predict.data.max(1, keepdim=True)[1]
                        predict=predict.squeeze()
                        reports = self.model.tokenizer.decode(predict.cpu().numpy())
                        temp = {'reports_ids': images_id[i], 'reports': reports,'label':'english'}
                        result_report_val_english.append(temp)
                        val_res_english.append(reports)
                        gtt=self.model.tokenizer.decode(reports_ids[i][1:].cpu().numpy())
                        ground_truths_eng.append(gtt)
                
                        val_gts_english=ground_truths_eng
                    else:
                        greedy_dec_input = self.greedy_decoder(self.model, report[i], reports_ids_use[i],start_symbol=2)
                        predict = self.model(report[i], greedy_dec_input)
                        predict=predict.data.max(1, keepdim=True)[1]
                   	    
                        predict=predict.squeeze()
                        reports = self.model.tokenizer.decode(predict.cpu().numpy())
                        reports1 = reports.replace(" ", "")
                        temp = {'reports_ids': images_id[i], 'reports': reports1,'label':'chinese'}
                        result_report_val_chinese.append(temp)
                        #reports=' '.join(reports)
                        val_res_chinese.append(reports)
                        gtt1=self.model.tokenizer.decode(reports_ids[i][1:].cpu().numpy())

                        #gtt1 = gtt1.replace(" ", "")
                        #gtt1 = ' '.join(gtt1)
                        ground_truths_cn.append(gtt1)
                
                        val_gts_chinese=ground_truths_cn

                    
                    
            val_met_english = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts_english)},
                                        {i: [re] for i, re in enumerate(val_res_english)})
            log.update(**{'val_english' + k: v for k, v in val_met_english.items()})
            resFiletest = 'valreports/english-' + str(epoch) + '.json'
            json.dump(result_report_val_english, open(resFiletest, 'w'))

            val_met_chinese = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts_chinese)},
                                       {i: [re] for i, re in enumerate(val_res_chinese)})
            log.update(**{'val_chinese' + k: v for k, v in val_met_chinese.items()})
            resFiletest = 'valreports/chinese-' + str(epoch) + '.json'
            json.dump(result_report_val_chinese, open(resFiletest, 'w', encoding='utf-8'),ensure_ascii=False)



        self.model.eval()
        with torch.no_grad():
            result_report_test_english = []
            result_report_test_chinese = []
            test_gts_english,test_gts_chinese, test_res_english, test_res_chinese = [], [], [] , []
            ground_truths_eng, ground_truths_cn=[], []

            for batch_idx, (images_id, images, reports_ids, report,image_path_all,reports_ids_use) in enumerate(self.test_dataloader):
                images, reports_ids,reports_ids_use= images.to(self.device), reports_ids.to(
                    self.device),reports_ids_use.to(self.device)



                for i in range(len(images_id)):
                    if  reports_ids[i][0]==1:
                        greedy_dec_input = self.greedy_decoder(self.model, report[i], reports_ids_use[i],start_symbol=1)
                        predict = self.model(report[i], greedy_dec_input)
                        predict=predict.data.max(1, keepdim=True)[1]
                   	    
                        predict=predict.squeeze()
                        reports = self.model.tokenizer.decode(predict.cpu().numpy())
                        temp = {'reports_ids': images_id[i], 'reports': reports,'label':'english'}
                        result_report_test_english.append(temp)
                        test_res_english.append(reports)
                        gtt2=self.model.tokenizer.decode(reports_ids[i][1:].cpu().numpy())
                        ground_truths_eng.append(gtt2)
                
                        test_gts_english=ground_truths_eng
                    else:
                        greedy_dec_input = self.greedy_decoder(self.model, report[i],reports_ids_use[i],start_symbol=2)
                        predict = self.model(report[i], greedy_dec_input)
                        predict=predict.data.max(1, keepdim=True)[1]
                   	    
                        predict=predict.squeeze()
                        reports = self.model.tokenizer.decode(predict.cpu().numpy())
                        reports1 = reports.replace(" ", "")
                        temp = {'reports_ids': images_id[i], 'reports': reports1,'label':'chinese'}
                        result_report_test_chinese.append(temp)
                        #reports = ' '.join(reports)
                        test_res_chinese.append(reports)
                        gtt3=self.model.tokenizer.decode(reports_ids[i][1:].cpu().numpy())
                        #gtt3 = gtt3.replace(" ", "")
                        #gtt3 = ' '.join(gtt3)
                        ground_truths_cn.append(gtt3)
                
                        test_gts_chinese=ground_truths_cn

                    
                    
            test_met_english = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts_english)},
                                        {i: [re] for i, re in enumerate(test_res_english)})
            log.update(**{'test_english' + k: v for k, v in  test_met_english.items()})
            resFiletest_eng = 'testreports/english-' + str(epoch) + '.json'
            json.dump(result_report_test_english, open(resFiletest_eng, 'w'))

            test_met_chinese = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts_chinese)},
                                       {i: [re] for i, re in enumerate(test_res_chinese)})
            log.update(**{'test_chinese' + k: v for k, v in test_met_chinese.items()})
            resFiletest = 'testreports/chinese-' + str(epoch) + '.json'
            json.dump(result_report_test_chinese, open(resFiletest, 'w', encoding='utf-8'),ensure_ascii=False)

        self.lr_scheduler.step()
        return log
        #self.lr_scheduler.step()

        #return log