import os
import sys

# 检查所有的的受试者的数据，也不是完全按照1-120的。去读文件夹的root_path = '/root/autodl-tmp/'，就知道有哪些个体
root_path = '/root/autodl-tmp/'  # 根据step3脚本中的root_path
output_root = 'ptt_output'
experiments = range(1, 12)  # exp_1 到 exp_11
# subject_ids = [50, 64, 82, 88, 89, 96, 97, 102]
# subject_list = [f'00{num:03d}' for num in subject_ids]
subject_list = os.listdir(root_path)
print(subject_list)


# 要检查的文件列表
expected_files = [
    'window_validation_exp_{exp}.csv',
    # 'valid_peaks_exp_{exp}.csv',
    # 'matched_heartbeats_windowed_exp_{exp}.csv',
    # 'ptt_windowed_exp_{exp}.csv',
    # 'ptt_summary_windowed_exp_{exp}.csv',
    'windowed_validation_exp_{exp}.png',
    # 'hr_validation_exp_{exp}.png'
]

def check_results():
    missing_files_report = {}
    
    for subject in subject_list:
        subject_output_dir = os.path.join(root_path, subject, output_root)
        if not os.path.exists(subject_output_dir):
            print(f'⚠️ 受试者 {subject} 的输出目录不存在: {subject_output_dir}')
            continue
        
        missing_files_report[subject] = {}
        for exp_id in experiments:
            exp_dir = os.path.join(subject_output_dir, f'exp_{exp_id}')
            if not os.path.exists(exp_dir):
                missing_files_report[subject][exp_id] = ['文件夹不存在']
                continue
            
            missing_in_exp = []
            for file_template in expected_files:
                file_name = file_template.format(exp=exp_id)
                file_path = os.path.join(exp_dir, file_name)
                if not os.path.exists(file_path):
                    missing_in_exp.append(file_name)
            
            if missing_in_exp:
                missing_files_report[subject][exp_id] = missing_in_exp
    
    # 打印报告
    print('\n📊 结果文件检查报告:')
    complete_subjects = []
    incomplete_subjects = []
    for subject, exp_missing in missing_files_report.items():
        if exp_missing:
            print(f'\n受试者 {subject}:')
            for exp_id, missing_list in exp_missing.items():
                print(f'  实验 {exp_id}: 缺失 {len(missing_list)} 个文件 - {missing_list}')
            incomplete_subjects.append(subject)
        else:
            print(f'✅ 受试者 {subject}: 所有实验文件完整')
            complete_subjects.append(subject)
    
    # 保存完整的受试者列表到TXT
    txt_path = '/root/PI_Lab/complete_subjects.txt'
    with open(txt_path, 'w') as f:
        for subj in sorted(complete_subjects):
            f.write(f'{subj}\n')
    print(f'\n💾 已保存完整的受试者列表到: {txt_path}')
    
    # 保存未完成的受试者列表到TXT
    incomplete_txt_path = '/root/PI_Lab/incomplete_subjects.txt'
    with open(incomplete_txt_path, 'w') as f:
        for subj in sorted(incomplete_subjects):
            f.write(f'{subj}\n')
    print(f'💾 已保存未完成的受试者列表到: {incomplete_txt_path}')

if __name__ == '__main__':
    check_results() 