import pandas as pd

class Config():
    weight_decay = 0
    lr = 1e-4
    opt = 'adam' #'lookahead_adam' to use `lookahead`
    momentum = 0.9
    model = "monologg/koelectra-base-v3-discriminator"#"beomi/kcbert-base"
    eval_metric = 'acc-score'
    clip_grad = 10.0
    clip_mode = "norm"
    log_interval = 50
    save_images = True
    device = 'cpu'#'cuda'
    epochs = 20
    sched = "cosine"
    min_lr = 1e-5
    warmup_lr = 0.0001
    warmup_epochs =3
    cooldown_epochs =10
    batch_size = 2 #100
    world_size = 1
    local_rank = 0
    path = '/content/output/'
    file_url = 'C:/Users/hhlee822/Desktop/시그로 코드/text_train_gpu/uploaded2.csv'
    num_classes = 2 #10034
    gamma = 1.5
    weight = None
    backbone_pretrained = True
    metric_score = ['f1-score', 'acc-score']
    fold = 5
    label_threshold = 5
    image_size = [256, 256]
    train_num = 4
    val_num = 4
    
    drop_rate = 0.3
    feature_dim = 768
    max_length = 128 
    early_stopping = 3
