class DefaultConfigs(object):
    train_data = "./moredata/" # where is your train data
    test_data = "./test/"   # your test data
    weights = "./checkpoints/"
    best_models = "./bestmodels/"
    submit = "./submit/"
    model_name = "fuck"
    num_classes = 28
    img_weight = 512
    img_height = 512
    channels = 4
    lr = 0.03
    batch_size = 32
    epochs = 50

config = DefaultConfigs()
