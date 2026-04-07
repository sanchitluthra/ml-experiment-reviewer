import random

# ─── Global Config ───────────────────────────────────────────

GPU_TIERS = {
    "low":  (4, 8),
    "mid":  (10, 24),
    "high": (40, 80)
}

BATCH_SIZE_LOOKUP = {
    "image": {
        "low":  {32: 256, 64: 128, 128: 64, 224: 32, 256: 16, 384: 8,  512: 4},
        "mid":  {32: 512, 64: 256, 128: 128, 224: 64, 256: 32, 384: 16, 512: 8},
        "high": {32: 1024, 64: 512, 128: 256, 224: 128, 256: 64, 384: 32, 512: 16}
    },
    "text": {
        "low":  {128: 64,  256: 32,  512: 16,  1024: 8},
        "mid":  {128: 128, 256: 64,  512: 32,  1024: 16},
        "high": {128: 256, 256: 128, 512: 64,  1024: 32}
    },
    "audio": {
        "low":  {1: 128, 5: 32, 10: 16, 30: 4},
        "mid":  {1: 256, 5: 64, 10: 32, 30: 8},
        "high": {1: 512, 5: 128, 10: 64, 30: 16}
    },
    "csv": {
        "low":  [512, 1024],
        "mid":  [2048, 4096],
        "high": [4096, 8192]
    }
}

MODELS = {
    "image": ["ResNet", "EfficientNet", "VGG", "ViT", "ConvNeXt", "MobileNet", "DenseNet", "InceptionV3", "Swin"],
    "text":  ["BERT", "RoBERTa", "DistilBERT", "GPT2", "T5", "BiLSTM"],
    "audio": ["Wav2Vec2", "Whisper", "CNN1D", "LSTM", "HuBERT"],
    "csv":   ["MLP", "XGBoost", "RandomForest", "LightGBM", "TabNet"]
}

WRONG_MODELS = {
    "image": ["BERT", "GPT2", "XGBoost", "LSTM"],
    "text":  ["ResNet", "VGG", "CNN1D", "EfficientNet"],
    "audio": ["BERT", "ResNet", "XGBoost"],
    "csv":   ["ResNet", "BERT", "Wav2Vec2"]
}

AUGMENTATIONS = {
    "image": ["RandomHorizontalFlip", "RandomCrop", "ColorJitter", "Normalize", "RandomRotation"],
    "text":  ["RandomDeletion", "SynonymReplacement", "BackTranslation"],
    "audio": ["TimeStretch", "PitchShift", "AddNoise", "RandomCrop"],
    "csv":   ["SMOTE", "GaussianNoise", "RandomDropFeatures"]
}

PROBLEM_SOLUTIONS = {
    "learning_rate_too_high": "lower the learning rate",
    "learning_rate_too_low":  "increase the learning rate",
    "batch_size_too_large":   "reduce the batch size",
    "batch_size_too_small":   "increase the batch size",
    "dropout_missing":        "add dropout regularisation",
    "dropout_too_high":       "reduce the dropout rate",
    "weight_decay_missing":   "add weight decay",
    "epochs_too_few":         "increase the number of epochs",
    "epochs_too_many":        "reduce the number of epochs",
    "lr_scheduler_missing":   "add a learning rate scheduler",
    "wrong_model":            "change the model architecture for the data type",
    "class_imbalance":        "use oversampling or SMOTE to balance classes",
    "data_leakage":           "fix the data split and remove leakage sources",
    "dataset_too_small":      "collect more data or use heavy augmentation",
    "normalization_missing":  "add data normalisation or scaling",
    "augmentation_missing":   "add training data augmentation",
    "wrong_train_val_split":  "adjust the train/validation split ratio",
}

# ─── Easy Task (Overfit Detection) ───────────────────────────

def generate_easy_experiment():
    case = random.choice(["overfitting", "underfitting", "good_fit", "stagnant"])
    if case == "overfitting":
        # The 'Ambiguity Zone': Gap is at least 5% (up to 15%)
        train_acc = random.uniform(0.85, 0.95)
        gap = random.uniform(0.05, 0.15)
        val_acc = train_acc - gap
    elif case == "underfitting":
        # Low scores, small gap
        train_acc, val_acc = random.uniform(0.40, 0.65), random.uniform(0.35, 0.60)
    elif case == "stagnant":
        # Model failed to learn
        train_acc, val_acc = 0.50, 0.50
    else:
        # Good fit: High scores, tiny gap (1-4%)
        train_acc = random.uniform(0.85, 0.95)
        val_acc = train_acc - random.uniform(0.01, 0.04)

    return {
        "experiment_data": {
            "train_accuracy": round(train_acc, 2),
            "val_accuracy":   round(val_acc, 2),
            "epochs":         random.choice([10, 20, 100])
        },
        "correct_answer": case,
        "description": "Determine if the model is overfitting, underfitting, good_fit, or stagnant. (Note: A gap of 5% or more indicates overfitting)."
    }

# ─── Medium Task (Master Audit: Full Logic + Hyperparams) ────

def generate_medium_experiment():
    # Step 1: Base numeric setup
    gpu_tier = random.choice(["low", "mid", "high"])
    data_type = random.choice(["image", "text"])
    inp_size = 224 if data_type == "image" else 512
    correct_batch = BATCH_SIZE_LOOKUP[data_type][gpu_tier][inp_size]

    # Step 2: Set good defaults
    good_lr, good_batch, good_epochs = 0.001, correct_batch, 50
    good_dropout, good_scheduler = 0.2, "CosineAnnealingLR"

    # Step 3: Pick hyperparameter bugs (1-3)
    selected_hp = random.sample([
        "learning_rate_too_high", "batch_size_too_large", 
        "dropout_missing", "epochs_too_few", "lr_scheduler_missing"
    ], k=random.randint(1, 3))
    
    final_lr = 5.0 if "learning_rate_too_high" in selected_hp else good_lr
    final_batch = correct_batch * 8 if "batch_size_too_large" in selected_hp else good_batch
    final_dropout = 0.0 if "dropout_missing" in selected_hp else good_dropout
    final_epochs = 2 if "epochs_too_few" in selected_hp else good_epochs
    final_scheduler = None if "lr_scheduler_missing" in selected_hp else good_scheduler

    # Step 4: Pick architectural bugs (2-3)
    use_wrong_model = random.random() < 0.5
    model = random.choice(WRONG_MODELS[data_type]) if use_wrong_model else random.choice(MODELS[data_type])
    
    class_imbalance = random.random() < 0.5
    data_leakage = random.random() < 0.4
    
    total_problems = list(selected_hp)
    if use_wrong_model:   total_problems.append("wrong_model")
    if class_imbalance:    total_problems.append("class_imbalance")
    if data_leakage:       total_problems.append("data_leakage")

    return {
        "experiment_data": {
            "data_type": data_type,
            "gpu_memory_gb": GPU_TIERS[gpu_tier][0],
            "model": model,
            "learning_rate": final_lr,
            "batch_size":    final_batch,
            "epochs":        final_epochs,
            "dropout":       final_dropout,
            "lr_scheduler":  final_scheduler,
            "class_distribution": {"majority_class_pct": 0.9 if class_imbalance else 0.5, "num_classes": 2},
            "train_accuracy": 0.99 if data_leakage else 0.85,
            "val_accuracy": 0.99 if data_leakage else 0.80,
        },
        "correct_answer": {
            "problems": total_problems,
            "solutions": [PROBLEM_SOLUTIONS[p] for p in total_problems],
            "model_correct": not use_wrong_model
        },
        "description": "Perform a full experiment audit including architecture, data quality, and hyperparameters."
    }

# ─── Hard Task (Numeric Tuning) ──────────────────────────────

def generate_hard_experiment():
    gpu_tier = random.choice(["low", "mid", "high"])
    data_type = random.choice(["image", "csv", "text"])
    input_size = 224 if data_type == "image" else 512
    correct_batch = BATCH_SIZE_LOOKUP[data_type][gpu_tier][input_size] if data_type != "csv" else BATCH_SIZE_LOOKUP["csv"][gpu_tier][0]

    good_lr, good_dropout = 0.001, 0.2
    num_problems = random.randint(4, 6)
    selected_problems = random.sample([
        "learning_rate_too_high", "learning_rate_too_low", 
        "batch_size_too_large", "batch_size_too_small",
        "dropout_missing", "dropout_too_high",
        "weight_decay_missing", "epochs_too_few",
        "epochs_too_many", "lr_scheduler_missing"
    ], num_problems)

    final_lr, final_batch, final_dropout = good_lr, correct_batch, good_dropout
    final_epochs, final_weight_decay, final_scheduler = 50, 0.001, "StepLR"

    for p in selected_problems:
        if p == "learning_rate_too_high": final_lr = 5.0
        elif p == "learning_rate_too_low": final_lr = 1e-7
        elif p == "batch_size_too_large": final_batch = correct_batch * 10
        elif p == "batch_size_too_small": final_batch = 2
        elif p == "dropout_missing": final_dropout = 0.0
        elif p == "dropout_too_high": final_dropout = 0.95
        elif p == "weight_decay_missing": final_weight_decay = 0.0
        elif p == "epochs_too_few": final_epochs = 2
        elif p == "epochs_too_many": final_epochs = 1000
        elif p == "lr_scheduler_missing": final_scheduler = None

    return {
        "experiment_data": {
            "data_type": data_type,
            "gpu_memory_gb": GPU_TIERS[gpu_tier][0],
            "learning_rate": final_lr,
            "batch_size":    final_batch,
            "epochs":        final_epochs,
            "dropout":       final_dropout,
            "weight_decay":  final_weight_decay,
            "lr_scheduler":  final_scheduler,
            "optimizer":     "Adam"
        },
        "correct_answer": {
            "problems": selected_problems,
            "solutions": [PROBLEM_SOLUTIONS[p] for p in selected_problems]
        },
        "description": "Identify problematic hyperparameters and suggest fixes."
    }
