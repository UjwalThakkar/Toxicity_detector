import torch
import transformers

MODEL_URLS = {
    "original":  "../checkpoint_files/toxic_original-c1212f89.ckpt",
}

PRETRAINED_MODEL = None


def get_model_and_tokenizer(model_type, model_name, tokenizer_name, num_classes, state_dict, huggingface_config_path=None):
    
    model_class = getattr(transformers, model_name)
       # model_class = <class 'transformers.models.bert.modeling_bert.BertForSequenceClassification'>

    config = model_class.config_class.from_pretrained(model_type, num_labels=num_classes)
    '''
        accessing the 'config_class' associated with the 'model_class'(BertForSequenceClassification)
        calling 'from_pretrained()' method from 'config_class' :
            from_pretrained takes 2 parameters: 
                1. model_type: This string specifies the type of pre-trained model you want to load the configuration from ('bert-base-uncased').
                2. num_labels (optional): This integer parameter allows you to adjust the pre-trained configuration for your specific task.
                    adjusts the 'num_labes' attribute to match the number of classes in our classification (6)
    '''

    model = model_class.from_pretrained(
        pretrained_model_name_or_path=None,
        config=huggingface_config_path or config,
        state_dict=state_dict,
        local_files_only=huggingface_config_path is not None,
    )

    tokenizer = getattr(transformers, tokenizer_name).from_pretrained(
        huggingface_config_path or model_type,
        local_files_only=huggingface_config_path is not None,
        # TODO: may be needed to let it work with Kaggle competition
        # model_max_length=512,
    )
    
    return model, tokenizer


def load_checkpoint(model_type="original", checkpoint=None, device="cpu", huggingface_config_path=None):
    if checkpoint is None:
        checkpoint_path = MODEL_URLS[model_type]
        loaded = torch.hub.load_state_dict_from_url(checkpoint_path, map_location=device)
    else:
        loaded = torch.load(checkpoint, map_location=device)
        if "config" not in loaded or "state_dict" not in loaded:
            raise ValueError(
                "Checkpoint needs to contain the config it was trained \
                    with as well as the state dict"
            )
    class_names = loaded["config"]["dataset"]["args"]["classes"]
     # standardise class names between models
    change_names = {
        "toxic": "toxicity",
        "identity_hate": "identity_attack",
        "severe_toxic": "severe_toxicity",
    }
    class_names = [change_names.get(cl, cl) for cl in class_names]
    model, tokenizer = get_model_and_tokenizer(
        **loaded['config']['arch']['args'],
        state_dict = loaded['state_dict'],
        huggingface_config_path = huggingface_config_path,
    )
    ''' 
    **loaded["config"]["arch"]["args"]
        - num_classes: 6
        - model_type: bert-base-uncased
        - model_name: BertForSequenceClassification
        - tokenizer_name: BertTokenizer
    '''
    
    return model, tokenizer, class_names


def load_model(model_type, checkpoint = None):
    if checkpoint is None:
        model, _, _,  = load_checkpoint(model_type = model_type)
    else:
        model, _, _, = load_checkpoint(checkpoint = checkpoint)
    return model


class toxicity_classifier:
    def __init__(self, model_type = "original", checkpoint = PRETRAINED_MODEL, device = "cpu", huggingface_config_path = None):
        super().__init__()
        self.model, self.tokenizer, self.class_names = load_checkpoint(
            model_type = model_type,
            checkpoint = checkpoint,
            device = device,
            huggingface_config_path = huggingface_config_path,   
        )
        self.device = device
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, text):
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        out = self.model(**inputs)[0]
        scores = torch.sigmoid(out).cpu().detach().numpy()
        results = {}
        for i, cla in enumerate(self.class_names):
            results[cla] = (
                scores[0][i] if isinstance(text, str) else [scores[ex_i][i].tolist() for ex_i in range(len(scores))]
            )
        return results

