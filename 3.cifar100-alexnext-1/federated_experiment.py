import os
import sys
import time
import torch
import yaml
import logging
import numpy.random

from datetime import datetime
from src.utils import get_duration
from src.server import CentralServer

if __name__ == '__main__':
    #torch.cuda.set_device(1)
    #os.environ['CUDA_VISIBLE_DEVICES']="1"
    ########################################################################################################
    '''
    file_path_member_norm = './log/Grad_Norm_Member.txt'  # 打开文件并写入文本内容 ,ke yi xiu gai wen jian ming yi bao cun
    file_path_nonmember_norm = './log/Grad_Norm_NonMember.txt'  # 打开文件并写入文本内容 ,ke yi xiu gai wen jian ming yi bao cun
    file_path_member_loss = './log/Loss_Member.txt'  # 打开文件并写入文本内容 ,ke yi xiu gai wen jian ming yi bao cun
    file_path_nonmember_loss = './log/Loss_NonMember.txt'  # 打开文件并写入文本内容 ,ke yi xiu gai wen jian ming yi bao cun

    with open(file_path_member_norm, 'a') as file:
        file.write("new start\n")

    with open(file_path_nonmember_norm, 'a') as file:
        file.write("new start\n")

    with open(file_path_member_loss, 'a') as file:
        file.write("new start\n")

    with open(file_path_nonmember_loss, 'a') as file:
        file.write("new start\n")
    '''
    ########################################################################################################


    # read input value
    #seed = int(sys.argv[1])
    seed=1

    # read configuration file
    # with open('./config_' + str(seed) + '.yaml') as c:
    with open('./config.yaml') as c:
        configs = list(yaml.load_all(c, Loader=yaml.FullLoader))

    experiment_config = configs[0]["experiment_config"]
    attack_config = configs[1]['attack_settings']
    training_config = configs[2]["training_settings"]
    data_config = configs[3]["data_settings"]
    log_config = configs[4]["LOG_settings"]
    backdoor_config = configs[5]["backdoor_settings"]

    # read persistent file to create separate log_files
    with open('./log/persistent.txt', 'r', encoding="utf-8") as f:
        read_data = int(f.read())
        file_number = str(read_data + 1)

    # write back current number + 1 for next run.
    with open('./log/persistent.txt', 'w', encoding="utf-8") as f:
        f.write(file_number)
    rootpath=experiment_config['model_path']

    experiment_config['model_path']=rootpath+'/global_model/'
    log_config['log_path']=rootpath+'/log/'
    attack_config['attack_model_path']=rootpath+'/attack_model/'
    imgpath=rootpath+'/img/'
    # create date string for unique file_naming
    date_string = log_config['log_path'] \
                  + str(datetime.now().month) + '_' \
                  + str(datetime.now().day) + '/'


    # create target model dir if not exists
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)

    # create target model dir if not exists
    if not os.path.exists(experiment_config['model_path']):
        os.makedirs(experiment_config['model_path'])

    # make directory per day for log files
    if not os.path.exists(imgpath):
        os.makedirs(imgpath)

    # make directory per day for log files
    if not os.path.exists(date_string):
        os.makedirs(date_string)

    # create target model dir if not exists
    if not os.path.exists(attack_config['attack_model_path']):
        os.makedirs(attack_config['attack_model_path'])

    # Create log filename
    log_filename = date_string + file_number
    ################
    log_filename_global=date_string +file_number+'_Global_'+str(seed)+'.txt'
    log_filename_local = date_string +file_number+'_Local_'+str(seed)+'.txt'
    log_filename_attack = date_string +file_number+ '_Attack_'+str(seed)+'.txt'

    log_filename_l2norm_member = date_string +file_number+ '_l2norm_member_'+str(seed)+'.txt'
    log_filename_l2norm_nonmember = date_string +file_number+ '_l2norm_nonmember_'+str(seed)+'.txt'
    log_filename_loss_member = date_string +file_number+ '_loss_member_'+str(seed)+'.txt'
    log_filename_loss_nomember = date_string +file_number+ '_loss_nomember_'+str(seed)+'.txt'



    log_global=logging.getLogger('log_global')
    log_local=logging.getLogger('log_local')
    log_attack=logging.getLogger('log_attack')

    log_l2norm_member=logging.getLogger('log_l2norm_member')
    log_l2norm_nonmember=logging.getLogger('log_l2norm_nonmember')
    log_loss_member=logging.getLogger('log_loss_member')
    log_loss_nomember=logging.getLogger('log_loss_nomember')



    file_handle_global=logging.FileHandler(log_filename_global)
    file_handle_local=logging.FileHandler(log_filename_local)
    file_handle_attack=logging.FileHandler(log_filename_attack)

    log_global.addHandler(file_handle_global)
    log_local.addHandler(file_handle_local)
    log_attack.addHandler(file_handle_attack)

    log_l2norm_member.addHandler(logging.FileHandler(log_filename_l2norm_member))
    log_l2norm_nonmember.addHandler(logging.FileHandler(log_filename_l2norm_nonmember))
    log_loss_member.addHandler(logging.FileHandler(log_filename_loss_member))
    log_loss_nomember.addHandler(logging.FileHandler(log_filename_loss_nomember))

    log_global.setLevel(logging.INFO)
    log_local.setLevel(logging.INFO)
    log_attack.setLevel(logging.INFO)
    log_l2norm_member.setLevel(logging.INFO)
    log_l2norm_nonmember.setLevel(logging.INFO)
    log_loss_member.setLevel(logging.INFO)
    log_loss_nomember.setLevel(logging.INFO)
    ################
    if experiment_config['train_model'] > 0:
        if 'active' in experiment_config['model_path']:
            log_filename += '_active_target_training_'
        else:
            log_filename += '_target_training_'

    elif attack_config['passive_attack'] > 0:
        if 'active' in experiment_config['model_path']:
            log_filename += '_active_attack_'
        else:
            log_filename += '_passive_attack_'
    # print("log_filename1",log_filename)

    # Set op logger
    logging.basicConfig(filename=log_filename + str(seed) + '.txt',
                        filemode='w',
                        level=getattr(logging, log_config['log_level']))

    # Setup device
    device = "cuda" if torch.cuda.is_available() and experiment_config['device'] == "cuda" else "cpu"

    # set seeds
    for iii in range(1):
        for seed in experiment_config['seed']:
            start_time = time.time()
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            numpy.random.seed(seed)

            # Setup Main Server with given parameters
            main_server = CentralServer(experiment_config,
                                        attack_config,
                                        data_config,
                                        training_config,
                                        backdoor_config,
                                        imgpath,
                                        rootpath
                                        )
            main_server.start_up()

            # Log start message.
            message = f"[ Experiment with SEED: {seed} started! ]"
            logging.info(msg=message)

            log_local = logging.getLogger('log_local')
            log_local.info(message)

            log_global = logging.getLogger('log_global')
            log_global.info(message)

            log_attack = logging.getLogger('log_attack')
            log_attack.info(message)
            # Perform main experiment
            main_server.perform_experiment()

            message = f"[ Total runtime: ... {str(get_duration(start_time))} for seed: {seed} ]"
            log_local = logging.getLogger('log_local')

            log_global = logging.getLogger('log_global')
            log_global.info(message)
            log_local.info(message)
            logging.info(message)
            log_attack = logging.getLogger('log_attack')
            log_attack.info(message)