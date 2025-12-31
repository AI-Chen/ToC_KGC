import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import OPTForCausalLM
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

class SentenceTransformer(nn.Module):
    def __init__(self,tokenizer_name='/srv/xiaoshuchen/MODEL/gpt2_base', model_name='/srv/xiaoshuchen/MODEL/gpt2_base',device='cpu'):
        super().__init__()
        self.device = device
        if "gpt2" in model_name:
            self.model_key = "gpt2"
            self.yidx, self.nidx, self.sep_idx, self.uidx = 3763, 645, 21017, 8627
        elif "Llama" in model_name:
            self.model_key = "llama"
            self.yidx, self.nidx = 3763, 645
        elif "opt" in model_name:
            self.model_key = "opt"
            self.yidx, self.nidx = 4420, 117
        if self.model_key == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        elif self.model_key == "llama":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.model = LlamaForCausalLM.from_pretrained(model_name).to(self.device)
        elif self.model_key == "opt":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.model = OPTForCausalLM.from_pretrained(model_name).to(self.device)

    def ce_loss(self, logits, label, task_label):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_label = label[..., 1:].contiguous()
        shift_task_label = task_label[..., 1:].contiguous()
        shift_label = shift_label.cuda(device=logits.device)
        shfit_task_label = shift_task_label.cuda(device=logits.device)
        task_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_label.reshape(-1))
        rationale_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_task_label.to(logits.device).view(-1))
        if torch.isnan(rationale_loss).any():
            rationale_loss = 0
        loss = 0.9 * task_loss + 0.1 * rationale_loss
        return loss

    def generate(self, prompt):
        tokens = self.tokenizer(prompt, return_tensors='pt')
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)
        output = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens= 512).squeeze()
        reason = self.tokenizer.decode(output, skip_special_tokens=True)
        return reason

    def forward(self, inputs, return_logits=False):
        if len(inputs) == 2:
            sentence, sentence_r = inputs
            encoded_input, encoded_input_r = self.tokenize(sentence, sentence_r)
        else:
            sentence, sentence_r = inputs, None
            encoded_input = self.tokenize(sentence, sentence_r)

        encoded_input = encoded_input.to(self.device)
        model_output = self.model(**encoded_input, return_dict=True)
        loss_a = model_output["loss"]
        logits = model_output["logits"]
        if sentence_r:
            loss_r_list = []
            for er in encoded_input_r:
                er = er.to(self.device)
                model_output = self.model(**er)
                loss_r_list.append(model_output["loss"])
            loss_r = sum(loss_r_list) / len(loss_r_list)
            loss = 0.5 * loss_a + 0.5 * loss_r
        else:
            loss = loss_a

        logits_yn = []
        for idx, ipt_id in enumerate(encoded_input["input_ids"]):
            ans_idx = encoded_input["attention_mask"][idx].sum() - 2
            logits_3 = torch.tensor([logits[idx][ans_idx][self.yidx].item(), logits[idx][ans_idx][self.uidx].item(), logits[idx][ans_idx][self.nidx].item()]).softmax(dim=0)
            logits_yn.append(2 * logits_3[0] + 1 * logits_3[1])
        logits_yes = torch.tensor(logits_yn)
        if return_logits:
            return loss, logits_yes
        return loss

    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def tokenize(self, sentences, sentences_r=None):
        tokens = self.tokenizer(list(sentences), padding=True, truncation=True, return_tensors='pt')
        labels = torch.empty(tokens['input_ids'].shape, dtype=tokens['input_ids'].dtype, device=tokens['input_ids'].device)
        if sentences_r:
            rnum = len(sentences_r[0])
            tokens_r_list = []
            labels_r_list = []
            sentences_r_reshape = [[] for _ in range(rnum)]
            for num in range(rnum):
                for rlist in sentences_r:
                    sentences_r_reshape[num].append(rlist[num])
            for num in range(rnum):
                tokens_r = self.tokenizer(sentences_r_reshape[num], padding=True, truncation=True, return_tensors='pt')
                tokens_r_list.append(tokens_r)
                labels_r_list.append(torch.empty(tokens_r['input_ids'].shape, dtype=tokens_r['input_ids'].dtype,
                                                 device=tokens_r['input_ids'].device))

        for idx, ipt_id in enumerate(tokens["input_ids"]):
            label = ipt_id.clone()
            ans_idx = tokens["attention_mask"][idx].sum() - 1
            label[: ans_idx] = -100
            label[label == self.tokenizer.pad_token_id] = -100
            labels[idx] = label
            if sentences_r:
                for jdx, tokens_r in enumerate(tokens_r_list):
                    label_r = tokens_r['input_ids'][idx].clone()
                    seq_length = label_r.shape[-1]
                    postion_ids = torch.tensor([i for i in range(seq_length)])
                    sep_idx = postion_ids[label_r == self.sep_idx][0]
                    label_r[:sep_idx + 1] = -100
                    label_r[label_r == self.tokenizer.pad_token_id] = -100
                    labels_r_list[jdx][idx] = label_r
        if sentences_r:
            for jdx, tokens_r in enumerate(tokens_r_list):
                tokens_r["labels"] = labels_r_list[jdx]
        tokens["labels"] = labels
        if sentences_r:
            return tokens, tokens_r_list
        else:
            return tokens
