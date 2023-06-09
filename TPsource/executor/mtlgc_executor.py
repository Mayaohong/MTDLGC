import os
import time
import numpy as np
import torch
from ray import tune
from TPsource.executor.traffic_state_executor import TrafficStateExecutor

class MTLGCExecutor(TrafficStateExecutor):
    def __init__(self, config, model, data_feature):
        TrafficStateExecutor.__init__(self, config, model, data_feature)
        self.K = config.get('K', 3)
        self.supports = data_feature.get('adj_mx')
        self.l1_lambda = self.config.get('l1_lambda', 0)
        self.l1_lambda = torch.tensor(self.l1_lambda).to(self.device)
        self.dataset = self.config.get('dataset', '')
        self.model_name = self.config.get('model', '')
        self.lgc_type = config.get('lgc_type', '')
        self.lgc = config.get('lgc', False)

    def train(self, train_dataloader, eval_dataloader):
        """
        use data to train model with config

        Args:
            train_dataloader(torch.Dataloader): Dataloader
            eval_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        train_time = []
        eval_time = []
        num_batches = len(train_dataloader)
        self._logger.info("num_batches:{}".format(num_batches))

        for epoch_idx in range(self.epochs):
            start_time = time.time()
            losses = self._train_epoch(train_dataloader, epoch_idx, self.loss_func)
            t1 = time.time()
            train_time.append(t1 - start_time)
            # 将测试损失和准确性写入SummaryWriter for tensorboard to analysis
            self._writer.add_scalar('training loss', np.mean(losses), epoch_idx)
            self._logger.info("epoch complete!")

            self._logger.info("evaluating now!")
            t2 = time.time()
            val_loss = self._valid_epoch(eval_dataloader, epoch_idx, self.loss_func)
            end_time = time.time()
            eval_time.append(end_time - t2)

            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            if (epoch_idx % self.log_every) == 0:
                log_lr = self.optimizer.param_groups[0]['lr']
                message = 'Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'.\
                    format(epoch_idx, self.epochs, np.mean(losses), val_loss, log_lr, (end_time - start_time))
                self._logger.info(message)

            if self.hyper_tune:
                # use ray tune to checkpoint
                with tune.checkpoint_dir(step=epoch_idx) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    self.save_model(path)
                # ray tune use loss to determine which params are best
                tune.report(loss=val_loss)

            if val_loss < min_val_loss:
                wait = 0
                if self.saved:
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break
        if len(train_time) > 0:
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), sum(train_time) / len(train_time),
                                     sum(eval_time) / len(eval_time)))
        if self.load_best_epoch:
            self.load_model_with_epoch(best_epoch)
        return min_val_loss

    def _train_epoch(self, train_dataloader, epoch_idx, loss_func=None):
        """
        完成模型一个轮次的训练

        Args:
            train_dataloader: 训练数据
            epoch_idx: 轮次数
            loss_func: 损失函数

        Returns:
            list: 每个batch的损失的数组
        """
        self.model.train()
        loss_func = loss_func if loss_func is not None else self.model.calculate_loss

        losses = []
        #save() 将数组保存到文件 
   
        for i, batch in enumerate(train_dataloader):
            #batch: (b,L,N,F)
            self.optimizer.zero_grad()
            batch.to_tensor(self.device)
            loss = loss_func(batch)
            
            #更新模型并获取当前时间步的邻接矩阵
            for i in range(self.K):
                lap = torch.tensor(self.supports[i]).type(torch.float32).to(self.device)
                non_zero_count = torch.numel(torch.nonzero(lap))
                l1_norm = torch.sum(torch.abs(lap))/non_zero_count # abs+sum up
                loss += self.l1_lambda[i] * l1_norm

            self._logger.debug(loss.item())
            losses.append(loss.item())
            loss.backward(retain_graph=True)  

            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        return losses

    def _valid_epoch(self, eval_dataloader, epoch_idx, loss_func=None):
        with torch.no_grad():
            self.model.eval()
            loss_func = loss_func if loss_func is not None else self.model.calculate_loss
            losses = []
            for i, batch in enumerate(eval_dataloader):
                batch.to_tensor(self.device)
                loss = loss_func(batch)
                self._logger.debug(loss.item())
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            self._writer.add_scalar('eval loss', mean_loss, epoch_idx)
            return mean_loss
