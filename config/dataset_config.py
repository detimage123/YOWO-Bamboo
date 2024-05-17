# Dataset configuration


dataset_config = {
    'ucf24': {
        # dataset
        'data_root': '/mnt/share/ssd2/dataset/STAD/ucf24',
        # 'data_root': 'D:/python_work/spatial-temporal_action_detection/dataset/ucf24',
        'gt_folder': './evaluator/groundtruths_ucf_jhmdb/groundtruths_ucf/',
        # input size
        'train_size': 224,
        'test_size': 224,
        # transform
        'pixel_mean': [0., 0., 0.],
        'pixel_std': [1., 1., 1.],
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5,
        'sampling_rate': 1,
        'len_clip': 16,
        # cls label
        'multi_hot': False,  # one hot
        # post process
        'conf_thresh': 0.3,
        'nms_thresh': 0.5,
        'conf_thresh_val': 0.005,
        'nms_thresh_val': 0.5,
        # freeze backbone
        'freeze_backbone_2d': False,
        'freeze_backbone_3d': False,
        # train config
        'batch_size': 8,
        'test_batch_size': 8,
        'accumulate': 16,
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'max_epoch': 5,
        'lr_epoch': [1, 2, 3, 4],
        'base_lr': 1e-4,
        'lr_decay_ratio': 0.5,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_iter': 500,
        # class names
        'valid_num_classes': 24,
        'label_map': (
                    'Basketball',     'BasketballDunk',    'Biking',            'CliffDiving',
                    'CricketBowling', 'Diving',            'Fencing',           'FloorGymnastics', 
                    'GolfSwing',      'HorseRiding',       'IceDancing',        'LongJump',
                    'PoleVault',      'RopeClimbing',      'SalsaSpin',         'SkateBoarding',
                    'Skiing',         'Skijet',            'SoccerJuggling',    'Surfing',
                    'TennisSwing',    'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog'
                ),
    },
    
    'jhmdb21': {
        # dataset
        'data_root': '/mnt/share/ssd2/dataset/STAD/jhmdb21',
        'data_root': 'D:/python_work/spatial-temporal_action_detection/dataset/jhmdb21',
        'gt_folder': './evaluator/groundtruths_ucf_jhmdb/groundtruths_jhmdb/',
        # input size
        'train_size': 224,
        'test_size': 224,
        # transform
        'pixel_mean': [0., 0., 0.],
        'pixel_std': [1., 1., 1.],
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5,
        'sampling_rate': 1,
        'len_clip': 16,
        # cls label
        'multi_hot': False,  # one hot
        # post process
        'conf_thresh': 0.3,
        'nms_thresh': 0.5,
        'conf_thresh_val': 0.005,
        'nms_thresh_val': 0.5,
        # freeze backbone
        'freeze_backbone_2d': True,
        'freeze_backbone_3d': True,
        # train config
        'batch_size': 8,
        'test_batch_size': 8,
        'accumulate': 16,
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'max_epoch': 5,
        'lr_epoch': [1, 2, 3, 4],
        'base_lr': 1e-4,
        'lr_decay_ratio': 0.5,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_iter': 500,
        # class names
        'label_map': (
                    'brush_hair',   'catch',          'clap',        'climb_stairs',
                    'golf',         'jump',           'kick_ball',   'pick', 
                    'pour',         'pullup',         'push',        'run',
                    'shoot_ball',   'shoot_bow',      'shoot_gun',   'sit',
                    'stand',        'swing_baseball', 'throw',       'walk',
                    'wave'
                ),
    },

    'ava_v2.2': {
        # dataset
        'data_root': '/home/4Tdisk/PBA1.0/AVA_Dataset',
        'frames_dir': 'frames/',
        'frame_list': 'frame_lists_yowo/',
        'annotation_dir': 'annotations_action+activate+pregnancy/',
        'train_gt_box_list': 'ava_train_v2.2.csv',
        'val_gt_box_list': 'ava_val_v2.2.csv',
        'train_exclusion_file': 'ava_train_excluded_timestamps_v2.2.csv',
        'val_exclusion_file': 'ava_val_excluded_timestamps_v2.2.csv',
        'labelmap_file': 'ava_action_list_v2.2_for_activitynet_2019.pbtxt',  # 'ava_v2.2/ava_action_list_v2.2.pbtxt',
        'class_ratio_file': 'config/panda_categories_ratio.json',
        'backup_dir': 'my_results/',
        # 'data_root': '/mnt/share/sda1/dataset/STAD/AVA_Dataset',
        # 'frames_dir': 'frames/',
        # 'frame_list': 'frame_lists/',
        # 'annotation_dir': 'annotations/',
        # 'train_gt_box_list': 'ava_v2.2/ava_train_v2.2.csv',
        # 'val_gt_box_list': 'ava_v2.2/ava_val_v2.2.csv',
        # 'train_exclusion_file': 'ava_v2.2/ava_train_excluded_timestamps_v2.2.csv',
        # 'val_exclusion_file': 'ava_v2.2/ava_val_excluded_timestamps_v2.2.csv',
        # 'labelmap_file': 'ava_v2.2/ava_action_list_v2.2_for_activitynet_2019.pbtxt', # 'ava_v2.2/ava_action_list_v2.2.pbtxt',
        # 'class_ratio_file': 'config/panda_categories_ratio.json',
        # 'backup_dir': 'results/',

        # input size
        'train_size': 224,
        'test_size': 224,
        # transform
        'pixel_mean': [0.45, 0.45, 0.45],
        'pixel_std': [0.225, 0.225, 0.225],
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5,
        'sampling_rate': 1,
        'len_clip': 16,
        # cls label
        'multi_hot': True,  # multi hot
        # post process
        'conf_thresh': 0.3,
        'nms_thresh': 0.5,
        'conf_thresh_val': 0.1,
        'nms_thresh_val': 0.5,
        # freeze backbone
        'freeze_backbone_2d': False,
        'freeze_backbone_3d': False,
        # train config
        'batch_size': 4,
        'test_batch_size': 8,
        'accumulate': 16,
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'max_epoch': 10,
        'lr_epoch': [3, 4, 5, 6],
        'base_lr': 1e-4,
        'lr_decay_ratio': 0.5,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_iter': 500,
        # class names
        'valid_num_classes': 22,
        'label_map': (
            "Continued lying", "Continued standing", "Continued sitting", "Continued prone", "Walking", "Climbing",
            "Rubbing", "Continued bipedal standing", "Drinking", "Eating", "Locomotive", "Resting", "Exploring",
            "Playing", "Marking", "Excretion", "Grooming","Pregnancy grasping", "Lick the chest and abdomen",
            "Lick the vulva", "Curl up", "Rub Yin"
        ),
    }
}
