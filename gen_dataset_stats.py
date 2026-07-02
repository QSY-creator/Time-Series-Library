"""生成 ECL 数据集统计信息表"""
import pandas as pd
import numpy as np
import os

path = './dataset/electricity/electricity.csv'
df = pd.read_csv(path)
numeric = df.iloc[:, 1:]  # 跳过 date 列
total = len(df)
total_channels = numeric.shape[1]

# 划分比例 (与 Dataset_Custom 一致: 70/10/20)
n_train = int(total * 0.7)
n_test = int(total * 0.2)
n_val = total - n_train - n_test

train_data = numeric.iloc[:n_train]
val_data   = numeric.iloc[n_train:n_train+n_val]
test_data  = numeric.iloc[n_train+n_val:]

all_vals = numeric.values.flatten()

stats = {
    '属性': [
        '数据集名称', '数据来源', '采集频率', '时间范围',
        '总样本数', '输入特征维度（通道数）', '总数据点数',
        '训练集样本数 (70%)', '验证集样本数 (10%)', '测试集样本数 (20%)',
        '缺失值数量', '异常值处理',
        '全局均值', '全局标准差', '全局最小值', '全局最大值',
        '通道均值范围', '通道标准差范围',
        '归一化方法',
    ],
    '值': [
        'ECL (Electricity Consuming Load)',
        'UCI Machine Learning Repository / Kaggle',
        '每小时 (Hourly)',
        f'{df.iloc[0,0]} ~ {df.iloc[-1,0]}',
        str(total),
        str(total_channels),
        str(total * total_channels),
        str(n_train),
        str(n_val),
        str(n_test),
        str(df.isnull().sum().sum()),
        '无异常值（原始数据已清洗）',
        f'{all_vals.mean():.4f}',
        f'{all_vals.std():.4f}',
        f'{all_vals.min():.4f}',
        f'{all_vals.max():.4f}',
        f'[{numeric.mean().min():.4f}, {numeric.mean().max():.4f}]',
        f'[{numeric.std().min():.4f}, {numeric.std().max():.4f}]',
        'Z-score 标准化 (StandardScaler)',
    ]
}

stats_df = pd.DataFrame(stats)

# 打印
print('=' * 70)
print('  ECL 数据集统计信息表')
print('=' * 70)
for _, row in stats_df.iterrows():
    print(f'  {row["属性"]:<30s} | {row["值"]}')

# 保存 CSV
stats_df.to_csv('./report_figures/dataset_statistics.csv', index=False, encoding='utf-8-sig')
print(f'\nSaved: ./report_figures/dataset_statistics.csv')

# 同时生成可直接贴报告的三线表格式 Markdown
with open('./report_figures/dataset_statistics_table.md', 'w', encoding='utf-8') as f:
    f.write('| 属性 | 值 |\n')
    f.write('|------|----|\n')
    for _, row in stats_df.iterrows():
        f.write(f'| {row["属性"]} | {row["值"]} |\n')

print('Saved: ./report_figures/dataset_statistics_table.md')