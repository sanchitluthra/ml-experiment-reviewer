import random

# ─── Medium Task Config ──────────────────────────────────────

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

# ─── Easy Task Generator ─────────────────────────────────────

def generate_easy_experiment():
    case = random.choice(["overfitting", "underfitting", "good_fit"])

    if case == "overfitting":
        train_acc  = random.uniform(0.92, 0.99)
        val_acc    = random.uniform(0.50, 0.72)
        train_loss = random.uniform(0.01, 0.05)
        val_loss   = random.uniform(0.70, 0.95)

    elif case == "underfitting":
        train_acc  = random.uniform(0.45, 0.68)
        val_acc    = random.uniform(0.40, 0.65)
        train_loss = random.uniform(0.40, 0.80)
        val_loss   = random.uniform(0.45, 0.85)

    elif case == "good_fit":
        base       = random.uniform(0.83, 0.92)
        train_acc  = base
        val_acc    = base - random.uniform(0.01, 0.04)
        train_loss = random.uniform(0.08, 0.15)
        val_loss   = train_loss + random.uniform(0.01, 0.04)

    return {
        "experiment_data": {
            "train_accuracy": round(train_acc, 2),
            "val_accuracy":   round(val_acc, 2),
            "train_loss":     round(train_loss, 2),
            "val_loss":       round(val_loss, 2),
            "epochs":         random.choice([10, 20, 30, 50, 100])
        },
        "correct_answer": case,
        "description": """You are a senior ML engineer reviewing an experiment.
        Analyze the training results and determine if the model is:
        - overfitting: train accuracy much higher than val accuracy
        - underfitting: both accuracies are low
        - good_fit: train and val accuracy are close and high
        Provide your diagnosis and reason."""
    }

# ─── Hard Task Generator (Pure Hyperparameter Tuning) ────────

def generate_hard_experiment():
    # Step 1: Pick GPU tier
    gpu_tier   = random.choices(["low", "mid", "high"], weights=[50, 35, 15])[0]
    gpu_memory = random.randint(GPU_TIERS[gpu_tier][0], GPU_TIERS[gpu_tier][1])

    # Step 2: Pick data type
    data_type = random.choices(["image", "csv", "text", "audio"], weights=[40, 30, 20, 10])[0]

    # Step 3: Pick input size
    if data_type == "image":
        input_size = random.choice([32, 64, 128, 224, 256, 384, 512])
    elif data_type == "text":
        input_size = random.choice([128, 256, 512, 1024])
    elif data_type == "audio":
        input_size = random.choice([1, 5, 10, 30])
    else:
        input_size = None

    # Step 4: Get correct batch size
    if data_type == "csv":
        correct_batch = random.choice(BATCH_SIZE_LOOKUP["csv"][gpu_tier])
    else:
        correct_batch = BATCH_SIZE_LOOKUP[data_type][gpu_tier][input_size]

    # Step 5: Generate good hyperparameters
    good_lr           = round(random.uniform(0.0001, 0.01), 5)
    good_epochs       = random.randint(10, 100)
    good_dropout      = round(random.uniform(0.1, 0.5), 2)
    good_weight_decay = round(random.uniform(0.0001, 0.01), 5)
    good_scheduler    = random.choice(["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"])
    good_optimizer    = random.choice(["Adam", "AdamW", "SGD"])

    # Step 6: Pick problems
    num_problems      = random.choices([2, 3, 4], weights=[40, 40, 20])[0]
    possible_problems = [
        "learning_rate_too_high", "learning_rate_too_low",
        "batch_size_too_large",   "batch_size_too_small",
        "dropout_missing",        "dropout_too_high",
        "weight_decay_missing",   "epochs_too_few",
        "epochs_too_many",        "lr_scheduler_missing",
    ]
    selected_problems = random.sample(possible_problems, num_problems)

    # Step 7: Apply problems
    final_lr           = good_lr
    final_batch        = correct_batch
    final_epochs       = good_epochs
    final_dropout      = good_dropout
    final_weight_decay = good_weight_decay
    final_scheduler    = good_scheduler

    for problem in selected_problems:
        if problem == "learning_rate_too_high":
            final_lr = round(random.uniform(0.5, 10.0), 3)
        elif problem == "learning_rate_too_low":
            final_lr = round(random.uniform(0.000001, 0.000009), 7)
        elif problem == "batch_size_too_large":
            final_batch = correct_batch * random.choice([4, 8, 16])
        elif problem == "batch_size_too_small":
            final_batch = random.choice([1, 2, 4])
        elif problem == "dropout_missing":
            final_dropout = 0.0
        elif problem == "dropout_too_high":
            final_dropout = round(random.uniform(0.8, 0.99), 2)
        elif problem == "weight_decay_missing":
            final_weight_decay = 0.0
        elif problem == "epochs_too_few":
            final_epochs = random.randint(1, 4)
        elif problem == "epochs_too_many":
            final_epochs = random.randint(300, 500)
        elif problem == "lr_scheduler_missing":
            final_scheduler = None

    return {
        "experiment_data": {
            "data_type":     data_type,
            "input_size":    input_size,
            "gpu_memory_gb": gpu_memory,
            "learning_rate": final_lr,
            "batch_size":    final_batch,
            "epochs":        final_epochs,
            "dropout":       final_dropout,
            "weight_decay":  final_weight_decay,
            "lr_scheduler":  final_scheduler,
            "optimizer":     good_optimizer,
        },
        "correct_answer": selected_problems,
        "description": """You are a senior ML engineer reviewing hyperparameters.
        Given the data type, input size, and GPU memory:
        - Identify which hyperparameters are incorrect
        - Explain why each one is problematic
        - Suggest correct values for each problem
        Consider relationships between GPU memory, input size, and batch size."""
    }

# ─── Medium Task Generator (Full Experiment Diagnosis) ───────

def generate_medium_experiment():
    # Step 1: Pick GPU
    gpu_tier   = random.choices(["low", "mid", "high"], weights=[50, 35, 15])[0]
    gpu_memory = random.randint(GPU_TIERS[gpu_tier][0], GPU_TIERS[gpu_tier][1])

    # Step 2: Pick data type
    data_type = random.choices(["image", "csv", "text", "audio"], weights=[40, 30, 20, 10])[0]

    # Step 3: Pick input size
    if data_type == "image":
        input_size = random.choice([32, 64, 128, 224, 256, 384, 512])
    elif data_type == "text":
        input_size = random.choice([128, 256, 512, 1024])
    elif data_type == "audio":
        input_size = random.choice([1, 5, 10, 30])
    else:
        input_size = None

    # Step 4: Pick dataset size
    dataset_size = random.choice([500, 1000, 5000, 10000, 50000, 100000])

    # Step 5: Pick model
    use_wrong_model = random.random() < 0.5
    if use_wrong_model:
        model         = random.choice(WRONG_MODELS[data_type])
        model_correct = False
    else:
        model         = random.choice(MODELS[data_type])
        model_correct = True

    # Step 6: Get correct batch size
    if data_type == "csv":
        correct_batch = random.choice(BATCH_SIZE_LOOKUP["csv"][gpu_tier])
    else:
        correct_batch = BATCH_SIZE_LOOKUP[data_type][gpu_tier][input_size]

    # Step 7: Generate good hyperparameters
    good_lr           = round(random.uniform(0.0001, 0.01), 5)
    good_epochs       = random.randint(10, 100)
    good_dropout      = round(random.uniform(0.1, 0.5), 2)
    good_weight_decay = round(random.uniform(0.0001, 0.01), 5)
    good_scheduler    = random.choice(["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"])
    good_optimizer    = random.choice(["Adam", "AdamW", "SGD"])

    # Step 8: No hyperparameter problems in Medium task
    selected_hp_problems = []

    # Apply good hyperparameters
    final_lr           = good_lr
    final_batch        = correct_batch
    final_epochs       = good_epochs
    final_dropout      = good_dropout
    final_weight_decay = good_weight_decay
    final_scheduler    = good_scheduler

    # Step 9: Data quality issues
    class_imbalance   = random.random() < 0.5
    data_leakage      = random.random() < 0.3
    dataset_too_small = dataset_size < 1000

    # Step 10: Preprocessing issues
    normalization_missing = random.random() < 0.5
    augmentation_missing  = random.random() < 0.5
    wrong_split           = random.random() < 0.3

    train_split = random.choice([0.5, 0.6]) if wrong_split else random.choice([0.7, 0.8])

    # Step 11: Build correct answers
    data_quality_issues  = []
    if class_imbalance:   data_quality_issues.append("class_imbalance")
    if data_leakage:      data_quality_issues.append("data_leakage")
    if dataset_too_small: data_quality_issues.append("dataset_too_small")

    preprocessing_issues = []
    if normalization_missing: preprocessing_issues.append("normalization_missing")
    if augmentation_missing:  preprocessing_issues.append("augmentation_missing")
    if wrong_split:           preprocessing_issues.append("wrong_train_val_split")

    # Build indirect signals instead of leaking boolean answers
    num_classes = random.choice([2, 5, 10, 20, 100])
    if class_imbalance:
        # Skewed distribution — agent must notice the imbalance
        majority_pct = random.uniform(0.70, 0.95)
        class_distribution = {
            "majority_class_pct": round(majority_pct, 2),
            "num_classes": num_classes,
            "smallest_class_samples": random.randint(5, 30),
        }
    else:
        class_distribution = {
            "majority_class_pct": round(1.0 / num_classes + random.uniform(-0.02, 0.02), 2),
            "num_classes": num_classes,
            "smallest_class_samples": max(50, dataset_size // (num_classes * 2)),
        }

    if data_leakage:
        # Suspicious perfect val accuracy signals leakage
        train_acc_reported = round(random.uniform(0.92, 0.98), 3)
        val_acc_reported   = round(random.uniform(0.96, 0.99), 3)  # val > train = red flag
    else:
        train_acc_reported = round(random.uniform(0.80, 0.92), 3)
        val_acc_reported   = round(train_acc_reported - random.uniform(0.02, 0.08), 3)

    # Preprocessing: show what was applied, not what's missing
    applied_transforms = []
    if not normalization_missing:
        applied_transforms.append("Normalize")
    if not augmentation_missing:
        applied_transforms.extend(random.sample(AUGMENTATIONS[data_type], min(2, len(AUGMENTATIONS[data_type]))))

    return {
        "experiment_data": {
            "data_type":          data_type,
            "input_size":         input_size,
            "dataset_size":       dataset_size,
            "gpu_memory_gb":      gpu_memory,
            "model":              model,
            "learning_rate":      final_lr,
            "batch_size":         final_batch,
            "epochs":             final_epochs,
            "dropout":            final_dropout,
            "weight_decay":       final_weight_decay,
            "lr_scheduler":       final_scheduler,
            "optimizer":          good_optimizer,
            "class_distribution": class_distribution,
            "train_accuracy":     train_acc_reported,
            "val_accuracy":       val_acc_reported,
            "applied_transforms": applied_transforms,
            "train_val_split":    train_split,
        },
        "correct_answer": {
            "hyperparameter_problems":   selected_hp_problems,
            "model_correct":             model_correct,
            "data_quality_issues":       data_quality_issues,
            "preprocessing_issues":      preprocessing_issues,
            "recommended_augmentations": AUGMENTATIONS[data_type],
        },
        "description": """You are a senior ML engineer doing a FULL experiment diagnosis.
        Analyze ALL aspects:
        1. Are the hyperparameters correct? (consider GPU memory and data type)
        2. Is the model correct for the data type?
        3. Are there data quality issues? (imbalance, leakage, size)
        4. Are preprocessing steps correct? (normalization, augmentation, split)
        5. What augmentations would you recommend?
        6. Give an overall experiment score from 0.0 to 1.0
        Be thorough and specific in your diagnosis."""
    }