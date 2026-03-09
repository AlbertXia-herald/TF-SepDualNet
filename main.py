# main.py 完整修复代码
import os
from lightning.pytorch.cli import LightningCLI
import torch
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import sys

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def analyze_domain_adaptation(log_dirs):
    import os
    import glob
    import pandas as pd
    import matplotlib.pyplot as plt

    results = []
    # 修复1：路径适配跨系统（使用os.path.sep）
    excluded_dirs = [os.path.join("log", "tfsepdualnet_train")]
    valid_log_dirs = [dir for dir in log_dirs if dir not in excluded_dirs]

    for base_dir in valid_log_dirs:
        metric_paths = glob.glob(os.path.join(base_dir, "version_*/metrics.csv"), recursive=True)
        if not metric_paths:
            print(f"警告：{base_dir} 下未找到metrics.csv，跳过")
            continue

        metric_path = sorted(metric_paths)[-1]  # 最新版本
        metrics = pd.read_csv(metric_path)

        # 修复2：准确筛选验证阶段行（包含source_acc且非空）
        valid_val_rows = metrics[metrics['source_acc'].notna() & (metrics['step'].notna())]
        if valid_val_rows.empty:
            print(f"警告：{metric_path} 无有效验证数据，跳过")
            continue

        # 修复3：正确提取训练损失（从训练阶段行获取）
        valid_train_rows = metrics[metrics['train/total_loss'].notna()]
        final_train_loss = valid_train_rows['train/total_loss'].iloc[-1] if not valid_train_rows.empty else 0.0

        last_val_epoch = valid_val_rows.iloc[-1]
        experiment_name = os.path.basename(base_dir)

        results.append({
            'experiment': experiment_name,
            'source_acc': last_val_epoch['source_acc'],
            'target_acc': last_val_epoch['target_acc'],
            'domain_gap': last_val_epoch['domain_gap'],
            'final_val_acc': last_val_epoch['val_acc'],
            'final_train_loss': final_train_loss  # 修复：从训练行提取
        })

    df = pd.DataFrame(results)
    if df.empty:
        print("未找到有效数据")
        return df

    # 修复4：动态调整域差距坐标轴范围
    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot(x='experiment', y=['source_acc', 'target_acc'], kind='bar', ax=ax,
            color=['#1f77b4', '#ff7f0e'], width=0.6)
    ax2 = ax.twinx()
    df.plot(x='experiment', y='domain_gap', kind='line', ax=ax2, color='red', marker='o', linewidth=2)

    ax.set_ylabel('Accuracy')
    ax2.set_ylabel('Domain Gap (Source - Target)', color='red')
    ax.set_title('Domain Adaptation Performance')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.set_ylim(0, 1)
    ax2.set_ylim(max(0, df['domain_gap'].min() - 0.01), df['domain_gap'].max() + 0.01)  # 动态范围
    plt.tight_layout()
    plt.savefig('domain_adaptation_comparison.png', dpi=300)

    return df
if __name__ == '__main__':
    # 检查是否使用分析模式
    if '--analyze' in sys.argv:
        log_dirs = glob("log/tfsepdualnet_*")  # 所有实验日志目录
        result_df = analyze_domain_adaptation(log_dirs)
        print(result_df)
    else:
        # 正常启动LightningCLI，保留原训练方式
        cli = LightningCLI()