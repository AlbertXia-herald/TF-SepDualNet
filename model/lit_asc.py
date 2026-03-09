from typing import Dict, Optional
import numpy as np
import torch
import torchinfo
import lightning as L
import torch.nn.functional as F
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from model.backbones import _BaseBackbone
from util.lr_scheduler import exp_warmup_linear_down
from util import _SpecExtractor, ClassificationSummary, _DataAugmentation
from torch.utils.tensorboard import SummaryWriter
import os

class LitAcousticSceneClassificationSystem(L.LightningModule):
    """
    Acoustic Scene Classification system based on LightningModule.
    Backbone model, data augmentation techniques and spectrogram extractor are designed to be plug-and-played.
    Backbone architecture, system complexity, classification report and confusion matrix are shown at test stage.

    Args:
        backbone (_BaseBackbone): Deep neural network backbone, e.g. cnn, transformer...
        data_augmentation (dict): A dictionary containing instances of data augmentation techniques in util/. Options: MixUp, FreqMixStyle, DeviceImpulseResponseAugmentation, SpecAugmentation. Set each to ``None`` if not use one of them.
        class_label (str): Class label. e.g. scene, device, city.
        domain_label (str): Domain label. e.g. scene, device, city.
        spec_extractor (_SpecExtractor): Spectrogram extractor used to transform 1D waveforms to 2D spectrogram. If ``None``, the input features should be 2D spectrogram.
    """

    def __init__(self,
                 backbone: _BaseBackbone,
                 data_augmentation: Dict[str, Optional[_DataAugmentation]],
                 class_label: str = "scene",
                 domain_label: str = "device",
                 spec_extractor: _SpecExtractor = None):
        super(LitAcousticSceneClassificationSystem, self).__init__()
        # Save the hyperparameters for Tensorboard visualization, 'backbone' and 'spec_extractor' are excluded.
        self.save_hyperparameters(ignore=['backbone', 'spec_extractor'])
        self.backbone = backbone
        self.data_aug = data_augmentation
        self.class_label = class_label
        self.domain_label = domain_label
        self.cla_summary = ClassificationSummary(class_label, domain_label)
        self.spec_extractor = spec_extractor

        # Save data during testing for statistical analysis
        self._test_step_outputs = {'emb': [], 'y': [], 'pred': [], 'd': []}
        # Input size of a 4D sample (1, 1, F, T), used for generating model profile.
        self._test_input_size = None

    @staticmethod
    def accuracy(logits, labels):
        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == labels).item() / len(labels)
        return acc, pred

    def forward(self, x, x_t=None):  # 新增 x_t 参数，默认 None
        """支持传递源域x和目标域x_t，适配域适应模型"""
        return self.backbone(x, x_t=x_t)  # 传递 x_t 到 backbone

    def training_step(self, batch, batch_idx):
        # Load a batch of waveforms with size (N, X)
        x = batch[0]
        # Store label dices in a dict
        labels = {'scene': batch[1], 'device': batch[2], 'city': batch[3]}
        # Choose class label
        y = labels[self.class_label]
        # Instantiate data augmentations
        dir_aug = self.data_aug[('dir_'
                                 'aug')]
        mix_style = self.data_aug['mix_style']
        spec_aug = self.data_aug['spec_aug']
        mix_up = self.data_aug['mix_up']
        # Apply dir augmentation on waveform
        x = dir_aug(x, labels['device']) if dir_aug is not None else x
        # Extract spectrogram from waveform

        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor is not None else x.unsqueeze(1)
        # Apply other augmentations on spectrogram
        x = mix_style(x) if mix_style is not None else x
        x = spec_aug(x) if spec_aug is not None else x
        if mix_up is not None:
            x, y = mix_up(x, y)
        # Get the predicted labels
        y_hat = self(x)
        # Calculate the loss and accuracy
        if mix_up is not None:
            pred = torch.argmax(y_hat, dim=1)
            train_loss = mix_up.lam * F.cross_entropy(y_hat, y[0]) + (1 - mix_up.lam) * F.cross_entropy(
                y_hat, y[1])
            corrects = (mix_up.lam * torch.sum(pred == y[0]) + (1 - mix_up.lam) * torch.sum(
                pred == y[1]))
            train_acc = corrects.item() / len(x)
        else:
            train_loss = F.cross_entropy(y_hat, y)
            train_acc, _ = self.accuracy(y_hat, y)
        # Log for each epoch
        self.log_dict({'train_loss': train_loss, 'train_acc': train_acc}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        labels = {'scene': batch[1], 'device': batch[2], 'city': batch[3]}
        y = labels[self.class_label]
        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor is not None else x.unsqueeze(1)
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        val_acc, _ = self.accuracy(y_hat, y)
        self.log_dict({'val_loss': val_loss, 'val_acc': val_acc}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_acc

    def test_step(self, batch, batch_idx):
        x = batch[0]
        labels = {'scene': batch[1], 'device': batch[2], 'city': batch[3]}
        y = labels[self.class_label]
        d = labels[self.domain_label]
        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor is not None else x.unsqueeze(1)
        # Get the input size of feature for measuring model profile
        self._test_input_size = (1, 1, x.size(-2), x.size(-1))
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        test_acc, pred = self.accuracy(y_hat, y)
        self.log_dict({'test_loss': test_loss, 'test_acc': test_acc})

        self._test_step_outputs['y'] += y.cpu().numpy().tolist()
        self._test_step_outputs['pred'] += pred.cpu().numpy().tolist()
        self._test_step_outputs['d'] += d.cpu().numpy().tolist()
        return test_acc

    def on_test_epoch_end(self):
        tensorboard = self.logger.experiment
        # Summary the model profile
        print("\n Model Profile:")
        if self._test_input_size is not None:
            # 生成符合 x_t 尺寸的假数据（根据 test_step 中 x_t 的输入格式）
            # 注意：input_size 是 (1, 1, freq, time)，对应 x_t 的形状
            dummy_x_t = torch.randn(*self._test_input_size).to(self.device)
            # 调用 summary 时，通过 kwargs 传入 forward 所需的全部参数
            model_profile = torchinfo.summary(
                self.backbone,
                args=[],  # 位置参数（若有）
                kwargs={
                    "x": None,  # 测试阶段 x 为 None（参考 test_step 中的调用）
                    "x_t": dummy_x_t,  # 假数据作为 x_t
                    "type": "test",  # 测试模式
                    "alpha": 0.0  # 测试阶段 alpha 固定为 0
                },
                device=self.device  # 确保设备匹配
            )
            # 记录模型概要
            tensorboard.add_text('model_profile', str(model_profile))
        macc = model_profile.total_mult_adds
        params = model_profile.total_params
        print('MACC:\t \t %.6f' % (macc / 1e6), 'M')
        print('Params:\t \t %.3f' % (params / 1e3), 'K\n')
        # Convert the summary to string
        model_summary = str(model_profile)
        model_summary += f'\n MACC:\t \t {macc / 1e6:.3f}M'
        model_summary += f'\n Params:\t \t {params / 1e3:.3f}K\n'
        model_summary = model_summary.replace('\n', '<br/>').replace(' ', '&nbsp;').replace('\t', '&emsp;')
        tensorboard.add_text('model_summary', model_summary)
        # Generate a classification report table
        tab_report = self.cla_summary.get_table_report(self._test_step_outputs)
        tensorboard.add_text('classification_report', tab_report)
        # Generate an confusion matrix figure
        cm = self.cla_summary.get_confusion_matrix(self._test_step_outputs)
        tensorboard.add_figure('confusion_matrix', cm)


class LitAscWithKnowledgeDistillation(LitAcousticSceneClassificationSystem):
    """
    ASC system with knowledge distillation.

    Args:
        temperature (float): A higher temperature indicates a softer distribution of pseudo-probabilities.
        kd_lambda (float): Weight to control the balance between kl loss and label loss.
        logits_index (int): Index of the logits in Dataset, as multiple logits may be used during training.
    """
    def __init__(self, temperature: float, kd_lambda: float, logits_index: int = -1, **kwargs):
        super(LitAscWithKnowledgeDistillation, self).__init__(**kwargs)
        self.temperature = temperature
        self.kd_lambda = kd_lambda
        self.logits_index = logits_index
        # KL Divergence loss for soft targets
        self.kl_div_loss = torch.nn.KLDivLoss(log_target=True)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        # Store label dices in a dict
        labels = {'scene': batch[1], 'device': batch[2], 'city': batch[3]}
        # Load soft labels
        teacher_logits = batch[self.logits_index]
        y_soft = F.log_softmax(teacher_logits / self.temperature, dim=-1)
        # Load hard labels
        y = labels[self.class_label]
        # Instantiate data augmentations
        dir_aug = self.data_aug['dir_aug']
        mix_style = self.data_aug['mix_style']
        spec_aug = self.data_aug['spec_aug']
        mix_up = self.data_aug['mix_up']
        # Apply dir augmentation on waveform
        x = dir_aug(x, labels['device']) if dir_aug is not None else x
        # Extract spectrogram from waveform
        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor is not None else x.unsqueeze(1)
        # Apply other augmentations on spectrogram
        x = mix_style(x) if mix_style is not None else x
        x = spec_aug(x) if spec_aug is not None else x
        if mix_up is not None:
            x, y, y_soft = mix_up(x, y, y_soft)
        # Get the predicted labels
        y_hat = self(x)
        # Temperature adjusted probabilities of teacher and student
        with torch.cuda.amp.autocast():
            y_hat_soft = F.log_softmax(y_hat / self.temperature, dim=-1)
        # Calculate the loss and accuracy
        if mix_up is not None:
            label_loss = mix_up.lam * F.cross_entropy(y_hat, y[0]) + (1 - mix_up.lam) * F.cross_entropy(y_hat, y[1])
            kd_loss = mix_up.lam * self.kl_div_loss(y_hat_soft, y_soft[0]) + (1 - mix_up.lam) * self.kl_div_loss(y_hat_soft, y_soft[1])
        else:
            label_loss = F.cross_entropy(y_hat, y)
            kd_loss = self.kl_div_loss(y_hat_soft, y_soft)
        kd_loss = kd_loss * (self.temperature ** 2)
        loss = self.kd_lambda * label_loss + (1 - self.kd_lambda) * kd_loss
        self.log_dict({'loss': loss, 'label_loss': label_loss, 'kd_loss': kd_loss}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


class LitAscWithWarmupLinearDownScheduler(LitAcousticSceneClassificationSystem):
    """
    ASC system with warmup-linear-down scheduler.
    """
    def __init__(self, optimizer: OptimizerCallable, warmup_len=4, down_len=26, min_lr=0.005, **kwargs):
        super(LitAscWithWarmupLinearDownScheduler, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.warmup_len = warmup_len
        self.down_len = down_len
        self.min_lr = min_lr

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        schedule_lambda = exp_warmup_linear_down(self.warmup_len, self.down_len, self.warmup_len, self.min_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class LitAscWithTwoSchedulers(LitAcousticSceneClassificationSystem):
    """
    ASC system with two customized schedulers.

    Directly instantiate multiple schedulers from the yaml config file.
    For more details: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced_3.html
    """
    def __init__(self, optimizer: OptimizerCallable, scheduler1: LRSchedulerCallable, scheduler2: LRSchedulerCallable, milestones, **kwargs):
        super(LitAscWithTwoSchedulers, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.milestones = milestones

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        scheduler1 = self.scheduler1(optimizer)
        scheduler2 = self.scheduler2(optimizer)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], self.milestones)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class LitAscWithThreeSchedulers(LitAcousticSceneClassificationSystem):
    """
    ASC system with three customized schedulers.

    Directly instantiate multiple schedulers from the yaml config file.
    For more details: https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced_3.html
    """
    def __init__(self, optimizer: OptimizerCallable, scheduler1: LRSchedulerCallable, scheduler2: LRSchedulerCallable, scheduler3: LRSchedulerCallable, milestones, **kwargs):
        super(LitAscWithThreeSchedulers, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.scheduler3 = scheduler3
        self.milestones = milestones

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        scheduler1 = self.scheduler1(optimizer)
        scheduler2 = self.scheduler2(optimizer)
        scheduler3 = self.scheduler3(optimizer)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2, scheduler3], self.milestones)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

class LitAscWithDomainAdaptationSystem(LitAcousticSceneClassificationSystem):
    """
    基于TFSepDualNet的声学场景分类域适应系统，支持源域与目标域联合训练，
    通过双分支对齐损失和域分类损失实现域自适应。
    """

    def __init__(self,
                 backbone: _BaseBackbone,
                 data_augmentation: Dict[str, Optional[_DataAugmentation]],
                 class_label: str = "scene",
                 domain_label: str = "device",  # 主域标签（设备），辅助域标签为城市
                 spec_extractor: _SpecExtractor = None,
                 domain_loss_weight: float = 1.0,  # 域分类损失权重（grl）
                 consistency_loss_weight: float = 0.1,  # 目标域一致性损失权重（mcd）
                 max_epochs: int = 150,  # 用于计算alpha的最大epoch数
                 optimizer: OptimizerCallable = None):  # 优化器（支持编码器和场景分类器分离）
        super().__init__(
            backbone=backbone,
            data_augmentation=data_augmentation,
            class_label=class_label,
            domain_label=domain_label,
            spec_extractor=spec_extractor
        )
        self.save_hyperparameters(ignore=['backbone', 'spec_extractor', 'optimizer'])
        # 域适应相关超参数
        self.domain_loss_weight = domain_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        self.max_epochs = max_epochs
        self.optimizer = optimizer

        # 存储测试阶段数据（用于后续分析）
        self._test_step_outputs = {'y': [], 'pred': [], 'd': []}
        self._test_input_size = None

    def _get_alpha(self, current_epoch, current_step, total_steps_per_epoch):
        """计算梯度反转层的alpha系数（随训练进程从0递增到1）"""
        p = (current_epoch * total_steps_per_epoch + current_step) / (self.max_epochs * total_steps_per_epoch)
        return 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0

    def training_step(self, batch, batch_idx):
        # 从AudioLabelsDatasetWithTarget解包数据：源域+目标域的波形和标签
        (wav_s, scene_label_s, device_label_s, city_label_s,
         wav_t, scene_label_t, device_label_t, city_label_t) = batch
        batch_size = wav_s.size(0)  # 单个域的batch大小

        # 1. 数据增强与特征提取（源域和目标域分别处理）
        # 1.1 源域波形增强（方向增强依赖设备标签）
        dir_aug = self.data_aug['dir_aug']
        x_s = dir_aug(wav_s, device_label_s) if dir_aug is not None else wav_s
        # 目标域波形增强（无需方向增强，或使用目标域设备标签）
        x_t = dir_aug(wav_t, device_label_t) if (dir_aug is not None) else wav_t

        # 1.2 提取频谱特征（添加通道维度）
        x_s = self.spec_extractor(x_s).unsqueeze(1) if self.spec_extractor else x_s.unsqueeze(1)
        x_t = self.spec_extractor(x_t).unsqueeze(1) if self.spec_extractor else x_t.unsqueeze(1)

        # 1.3 频谱增强（混合风格、频谱裁剪）
        mix_style = self.data_aug['mix_style']
        spec_aug = self.data_aug['spec_aug']
        x_s = mix_style(x_s) if mix_style is not None else x_s
        x_s = spec_aug(x_s) if spec_aug is not None else x_s
        x_t = mix_style(x_t) if mix_style is not None else x_t
        x_t = spec_aug(x_t) if spec_aug is not None else x_t

        # 2. 计算当前alpha（梯度反转系数）
        total_steps_per_epoch = len(self.trainer.train_dataloader)
        alpha = self._get_alpha(self.current_epoch, batch_idx, total_steps_per_epoch)

        # 3. TFSepDualNet前向传播（训练模式）
        # 输出：设备预测、城市预测、源域双场景头、目标域双场景头（已softmax）
        pre_device, pre_city, f1_src, f2_src, t1_tgt, t2_tgt = self.backbone(
            x=x_s, x_t=x_t, type="train", alpha=alpha
        )

        # 4. 损失计算
        # 4.1 源域场景分类损失（双分支平均）
        scene_loss = (F.cross_entropy(f1_src, scene_label_s) +
                      F.cross_entropy(f2_src, scene_label_s)) * 0.5

        # 4.2 域分类损失（设备+城市，源域权重更高）
        # 设备损失：源域权重0.5，目标域权重1.0
        device_loss = (0.5 * F.cross_entropy(pre_device[:batch_size], device_label_s) +
                       F.cross_entropy(pre_device[batch_size:], device_label_t)) / 1.5
        # 城市损失：源域权重1.0，目标域权重0.25
        city_loss = (F.cross_entropy(pre_city[:batch_size], city_label_s) +
                     0.25 * F.cross_entropy(pre_city[batch_size:], city_label_t)) / 1.25
        domain_total_loss = device_loss + city_loss

        # 4.3 目标域一致性损失（MCD：双分支L1距离）
        consistency_loss = torch.mean(torch.sum(torch.abs(t1_tgt - t2_tgt), dim=-1))

        # 4.4 总损失（加权求和）
        total_loss = (scene_loss +
                      self.domain_loss_weight * domain_total_loss +
                      self.consistency_loss_weight * consistency_loss)

        # 5. 计算准确率（日志监控）
        _, pred_scene = self.accuracy(f1_src + f2_src, scene_label_s)  # 源域场景准确率
        _, pred_device = self.accuracy(pre_device, torch.cat([device_label_s, device_label_t]))  # 设备域准确率
        _, pred_city = self.accuracy(pre_city, torch.cat([city_label_s, city_label_t]))  # 城市域准确率

        # 6. 日志记录
        self.log_dict({
            'train/total_loss': total_loss,
            'train/scene_loss': scene_loss,
            'train/domain_loss': domain_total_loss,
            'train/consistency_loss': consistency_loss,
            'train/scene_acc': torch.sum(pred_scene == scene_label_s).item() / len(scene_label_s),
            'train/device_acc': torch.sum(pred_device == torch.cat([device_label_s, device_label_t])).item() / (
                        2 * batch_size),
            'train/city_acc': torch.sum(pred_city == torch.cat([city_label_s, city_label_t])).item() / (2 * batch_size),
            'train/alpha': alpha
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """验证步骤：使用源域数据验证场景分类性能"""
        x = batch[0]  # 源域波形
        labels = {'scene': batch[1], 'device': batch[2], 'city': batch[3]}
        y = labels[self.class_label]

        # 特征提取（无增强）
        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor else x.unsqueeze(1)
        # 模型前向（验证模式：返回双分支融合结果）
        y_hat = self.backbone(x=None, x_t=x, type="valid", alpha=0.0)
        val_loss = F.cross_entropy(y_hat, y)
        val_acc, _ = self.accuracy(y_hat, y)

        self.log_dict({'val_loss': val_loss, 'val_acc': val_acc},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_acc

    def test_step(self, batch, batch_idx):
        """测试步骤：使用目标域数据评估最终性能"""
        x = batch[0]  # 目标域波形
        labels = {'scene': batch[1], 'device': batch[2], 'city': batch[3]}
        y = labels[self.class_label]
        d = labels[self.domain_label]  # 域标签（用于后续按域分析）

        # 特征提取
        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor else x.unsqueeze(1)
        self._test_input_size = (1, 1, x.size(-2), x.size(-1))  # 模型输入尺寸记录
        # 模型前向（测试模式：双分支融合）
        y_hat = self.backbone(x=None, x_t=x, type="test", alpha=0.0)
        test_loss = F.cross_entropy(y_hat, y)
        test_acc, pred = self.accuracy(y_hat, y)

        # 存储测试结果（用于 epoch 结束时生成报告）
        self._test_step_outputs['y'] += y.cpu().numpy().tolist()
        self._test_step_outputs['pred'] += pred.cpu().numpy().tolist()
        self._test_step_outputs['d'] += d.cpu().numpy().tolist()
        self.log_dict({'test_loss': test_loss, 'test_acc': test_acc})
        return test_acc

    def configure_optimizers(self):
        """配置优化器：支持编码器和场景分类器分离优化（参考train.py的双优化器逻辑）"""
        if hasattr(self.backbone, 'encoder') and hasattr(self.backbone, 'scene'):
            # 分离编码器和场景分类器参数（分别优化）
            encoder_params = list(self.backbone.encoder.parameters())
            scene_params = list(self.backbone.scene.parameters())
            optimizer = self.optimizer([
                {'params': encoder_params, 'lr': self.hparams.lr},
                {'params': scene_params, 'lr': self.hparams.lr}
            ])
        else:
            # 整体优化
            optimizer = self.optimizer(self.parameters())
        return optimizer
