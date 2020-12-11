# coding=utf-8

###### 欢迎使用脚本任务,让我们首选熟悉下一些使用规则吧 ###### 

# 数据集文件目录
datasets_prefix = '/root/paddlejob/workspace/train_data/datasets/'

# 数据集文件具体路径请在编辑项目状态下,通过左侧导航栏「数据集」中文件路径拷贝按钮获取
train_datasets =  '通过路径拷贝获取真实数据集文件路径 '

# 输出文件目录. 任务完成后平台会自动把该目录所有文件压缩为tar.gz包，用户可以通过「下载输出」可以将输出信息下载到本地.
output_dir = "/root/paddlejob/workspace/output"

# 日志记录. 任务会自动记录环境初始化日志、任务执行日志、错误日志、执行脚本中所有标准输出和标准出错流(例如print()),用户可以在「提交」任务后,通过「查看日志」追踪日志信息

import os

if __name__ == '__main__':
    
    print(os.getcwd())
    print("预装依赖包")
    os.system("pip install -i https://mirror.baidu.com/pypi/simple --upgrade pip")
    os.system("pip install -i https://mirror.baidu.com/pypi/simple sentencepiece")
    print("解压Knover模块")
    #os.system("unzip /root/paddlejob/workspace/train_data/datasets/data56424/Knover.zip")
    os.system("unzip /root/paddlejob/workspace/train_data/datasets/data57647/Knover.zip")
    os.system("unzip /root/paddlejob/workspace/train_data/datasets/data57647/model.zip")
    os.system("unzip /root/paddlejob/workspace/train_data/datasets/data57647/NSP.zip")
    os.system("unzip /root/paddlejob/workspace/train_data/datasets/data57647/test_2.zip")
    print("解压数据集")
    #os.system("unzip /root/paddlejob/workspace/train_data/datasets/data56424/pro_data.zip")
    
    print("开始训练")
    os.system("export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7")
    os.system("cd ./home/aistudio/Knover/latest_model/ && ls -a")
    
    #####################################################PARAMETERS######################################################
    epochs = 1
    start_step = 0
    lr = 1e-4
    model = "Plato"
    batch_size = 1
    model_name = "pt_model"  # {"ut_model": UnifiedTransformer, "pt_model": Plato}
    in_tokens = True  # infer.py 中设为False
    
    config_name = "12L_P.json"  # {"12L.json", "12L_P.json", "24L.json", "24L_P.json"}, P is for Plato model
    
    func_py = "infer.py"  # {"train.py", "infer.py"}
    
    split_size = 5000  # infer.py 运行时切分文件所含样本最大数量
    ####################################################################################################################
    
    if func_py == 'train.py':
        if model == 'Plato' or model == 'UnifiedTransformer':
            args = "--model {} --task DialogGeneration --vocab_path ./home/aistudio/Knover/config/vocab.txt --spm_model_file ./home/aistudio/Knover/config/spm.model \
                --train_file ./home/aistudio/pro_data/train.txt --valid_file ./home/aistudio/pro_data/valid.txt --data_format numerical --file_format file --config_path ./home/aistudio/Knover/config/{} \
                --in_tokens {} --batch_size {} -lr {} --warmup_steps 1000 --weight_decay 0.01 --num_epochs {} \
                --max_src_len 384 --max_tgt_len 128 --max_seq_len 512 \
                --log_step 100 --validation_steps 5000 --save_steps 100 \
                --is_distributed True is_cn True --start_step {} \
                --init_checkpoint ./model/{} \
                --save_path /root/paddlejob/workspace/output \
                ".format(model, config_name, in_tokens, batch_size, lr, epochs, start_step, model_name)
            os.system("python -m paddle.distributed.launch ./home/aistudio/Knover/{} {}".format(func_py, args))
        elif model == 'NSPModel':
            args = "--model {} --task NextSentencePrediction --vocab_path ./home/aistudio/Knover/config/vocab.txt --spm_model_file ./home/aistudio/Knover/config/spm.model \
                --train_file ./home/aistudio/pro_data/train.txt --valid_file ./home/aistudio/pro_data/valid.txt --data_format numerical --file_format file --config_path ./home/aistudio/Knover/config/{} \
                --in_tokens {} --batch_size {} -lr {} --warmup_steps 1000 --weight_decay 0.01 --num_epochs {} \
                --max_src_len 384 --max_tgt_len 128 --max_seq_len 512 \
                --log_step 100 --validation_steps 5000 --save_steps 100 \
                --is_distributed True --start_step {} \
                --init_checkpoint ./model/{} \
                --save_path /root/paddlejob/workspace/output \
                --mix_negative_sample True \
                ".format(model, config_name, in_tokens, batch_size, lr, epochs, start_step, model_name)
            os.system("python -m paddle.distributed.launch ./home/aistudio/Knover/{} {}".format(func_py, args))
        else:
            raise ValueError("Only support Plato, UnifiedTransformer, and NSPModel but received %s" % model)
