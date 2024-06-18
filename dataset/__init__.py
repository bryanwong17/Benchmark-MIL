def get_default_CLASS_LABELS():
    CLASS_NAMES = ['0', '1']
    return CLASS_NAMES

def get_camelyon16_CLASS_LABELS():
    CLASS_NAMES = ['normal', 'tumor']
    return CLASS_NAMES

def get_tcga_nsclc_CLASS_LABELS():
    CLASS_NAMES = ['LUAD', 'LUSC']
    return CLASS_NAMES

def get_seegene_old_CLASS_LABELS():
    CLASS_NAMES = ['D', 'M', 'N']
    return CLASS_NAMES

def get_seegene_new_CLASS_LABELS():
    CLASS_NAMES = ['D', 'M', 'N', 'P', 'R', 'S', 'V'] # CHANGE
    return CLASS_NAMES

def get_patch_gastric_adc_22_CLASS_LABELS(): # only uses 3 majority classes out of 9 classes
    CLASS_NAMES = ['0', '1', '2']
    return CLASS_NAMES

def get_class_names(dataset_name):
    if dataset_name == "camelyon16":
        class_names = get_camelyon16_CLASS_LABELS()
    elif dataset_name == "tcga_nsclc":
        class_names = get_tcga_nsclc_CLASS_LABELS()
    elif dataset_name == "seegene_old":
        class_names = get_seegene_old_CLASS_LABELS()
    elif dataset_name == "seegene_new":
        class_names = get_seegene_new_CLASS_LABELS()
    elif dataset_name == "patch_gastric_adc_22":
        class_names = get_patch_gastric_adc_22_CLASS_LABELS()
    elif dataset_name is None:
        print("Not specify dataset, use default dataset with label 0, 1 instead.")
        class_names = get_default_CLASS_LABELS()
    else:
        raise NotImplementedError

    return class_names
