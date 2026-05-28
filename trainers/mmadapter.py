from collections import OrderedDict
from collections import defaultdict
import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.utils import load_pretrained_weights, load_checkpoint
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler
import copy
from clip import clip

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model_MMA(state_dict or model.state_dict(), cfg)
    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_rep_tokens_text, retrun_adapater_func=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if retrun_adapater_func == None:
            x = self.transformer(x)
        else:
            x, x2 = self.transformer([x, x, compound_rep_tokens_text, retrun_adapater_func, 0])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x2 = x2.permute(1, 0, 2)  # LND -> NLD
        x2 = self.ln_final(x2).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        x2 = x2[torch.arange(x2.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x, x2

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Adapter(nn.Module):
    def __init__(self, in_dim, out_dim, bottleneck=32):
        super().__init__()
        self.down = nn.Linear(in_dim, bottleneck)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(bottleneck, out_dim)

        # Initialization
        nn.init.kaiming_normal_(self.down.weight)
        nn.init.zeros_(self.down.bias)

        nn.init.zeros_(self.up.weight)   # 🔑 makes adapter initially inactive
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.act(self.down(x)))

class AdapterLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        self.n_cls = len(classnames)
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self._build_text_embedding(cfg, classnames, clip_model)

        # build multi-modal adapter
        self.text_adapter_func = lambda x: self.return_text_adapter(index=x)
        self.text_adapter = self._build_adapter(
            clip_model.ln_final.weight.shape[0], 
            len(clip_model.transformer.resblocks), 
            cfg.TRAINER.MMADAPTER.ADAPTER_LAYERS,
            cfg.TRAINER.MMADAPTER.ADAPTER_DIM,
            clip_model.dtype
        )
        
        self.visual_adapter_func = lambda x: self.return_visual_adapter(index=x)
        self.visual_adapter = self._build_adapter(
            clip_model.visual.ln_post.weight.shape[0],
            len(clip_model.visual.transformer.resblocks), 
            cfg.TRAINER.MMADAPTER.ADAPTER_LAYERS,
            cfg.TRAINER.MMADAPTER.ADAPTER_DIM,
            clip_model.dtype,
            visual = True
        )

        if cfg.TRAINER.MMADAPTER.IS_SHARED:
            self.shared_adapter = self._build_adapter(
                cfg.TRAINER.MMADAPTER.ADAPTER_DIM,
                len(clip_model.visual.transformer.resblocks), 
                cfg.TRAINER.MMADAPTER.ADAPTER_LAYERS,
                cfg.TRAINER.MMADAPTER.ADAPTER_DIM,
                clip_model.dtype
            )

            self.shared_adapter_gen = self._build_adapter(
                cfg.TRAINER.MMADAPTER.ADAPTER_DIM,
                len(clip_model.visual.transformer.resblocks), 
                cfg.TRAINER.MMADAPTER.ADAPTER_LAYERS,
                cfg.TRAINER.MMADAPTER.ADAPTER_DIM,
                clip_model.dtype
            )
        else:
            adapter = [None] * (len(clip_model.visual.transformer.resblocks) + 1)
            adapter = nn.ModuleList([a for a in adapter])
            self.shared_adapter = adapter

        self.adapter_scale = float(cfg.TRAINER.MMADAPTER.ADAPTER_SCALE)

        n_rep_tokens = 1
        rep_dim = 128
        text_dim = clip_model.ln_final.weight.shape[0]
        visual_dim = clip_model.visual.ln_post.weight.shape[0]
        self.rep_layers_length = len(cfg.TRAINER.MMADAPTER.ADAPTER_LAYERS)
        self.dtype = clip_model.dtype

        # self.compound_rep_tokens = nn.Parameter(torch.empty(n_rep_tokens, rep_dim))
        # nn.init.normal_(self.compound_rep_tokens, std=0.02)
        self.compound_rep_tokens = nn.Parameter(torch.empty(n_rep_tokens, rep_dim))
        nn.init.kaiming_normal_(self.compound_rep_tokens, a=0, mode='fan_in', nonlinearity='relu')

        # single_layer_r2v = nn.Linear(rep_dim, visual_dim)
        # single_layer_r2t = nn.Linear(rep_dim, text_dim)


        single_layer_r2v = Adapter(rep_dim, visual_dim, bottleneck=32)
        single_layer_r2t = Adapter(rep_dim, text_dim, bottleneck=2)
    #4599040

        self.compound_rep_tokens_r2vproj  = _get_clones(single_layer_r2v, self.rep_layers_length)
        self.compound_rep_tokens_r2tproj = _get_clones(single_layer_r2t, self.rep_layers_length)


    def return_text_adapter(self, index):
        return self.text_adapter[index], self.shared_adapter[index], self.shared_adapter_gen[12], self.adapter_scale

    def return_visual_adapter(self, index):
        return self.visual_adapter[index], self.shared_adapter[index], self.shared_adapter_gen[12], self.adapter_scale


    def _build_text_embedding(self, cfg, classnames, clip_model):
        dtype = clip_model.dtype
        text_ctx_init = cfg.TRAINER.MMADAPTER.TEXT_CTX_INIT

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [text_ctx_init + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_embedding", embedding)
        self.register_buffer("tokenized_prompts", tokenized_prompts)


    def _build_adapter(self, d_model, n_layers, adapter_layers, mid_dim, dtype, visual=False):

        adapter = [None] * (n_layers + 1)
        for i in adapter_layers:
            if mid_dim == d_model:
                adapter[i] = nn.Sequential(
                    nn.Linear(d_model, mid_dim),
                    nn.ReLU()
                )
            else:
                adapter[i] = nn.Sequential(OrderedDict([
                    ("down", nn.Sequential(nn.Linear(d_model, mid_dim), nn.ReLU())),
                    ("up", nn.Linear(mid_dim, d_model))
                ]))
        adapter = nn.ModuleList([a for a in adapter])
        for m in adapter.modules():
            if isinstance(m, nn.Linear):
                # if visual:
                #     nn.init.zeros_(m.weight)
                #     if m.bias is not None:
                #         nn.init.zeros_(m.bias)
                # else:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

        if dtype == torch.float16:
            for m in adapter.modules():
                m.half()
    
        return adapter
    
    def forward(self):
        embedding = self.token_embedding
        if self.text_adapter[0] is not None:
            token_embedding = self.text_adapter[0].down(embedding)
            shared_adapter = self.shared_adapter[0]
            token_embedding = shared_adapter(token_embedding)
            token_embedding = self.text_adapter[0].up(token_embedding)
            embedding = embedding + self.adapter_scale * token_embedding

        
        compound_rep_tokens_visual = []
        compound_rep_tokens_text = []
 
        for index in range(self.rep_layers_length):
            rep_tokens = self.compound_rep_tokens
            rep_mapped_to_text = self.compound_rep_tokens_r2tproj[index](rep_tokens)
            rep_mapped_to_visual = self.compound_rep_tokens_r2vproj[index](rep_tokens)                        
            compound_rep_tokens_text.append(rep_mapped_to_text.type(self.dtype))
            compound_rep_tokens_visual.append(rep_mapped_to_visual.type(self.dtype))                         


        return embedding, self.text_adapter_func, self.visual_adapter_func, compound_rep_tokens_text, compound_rep_tokens_visual

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        self.adapter_learner = AdapterLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.adapter_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.text_features_for_inference = None
        self.per_text_features_for_inference = None
        self.all_features = []
        self.all_features_text = []
        self.Lambda = cfg.TRAINER.MMADAPTER.LAMBDA

    def encode_text(self, prompts, tokenized_prompts, compound_rep_tokens_text, text_adapter_func=None):
        if text_adapter_func is not None:
            text_features, per_text_features = self.text_encoder(
                prompts, tokenized_prompts, compound_rep_tokens_text, text_adapter_func
            )
        else:
            text_features = self.text_encoder(
                prompts, tokenized_prompts, compound_rep_tokens_text
            )
        return text_features, per_text_features
    
    def encode_image(self, image, compound_rep_tokens_visual, visual_adapter_func=None):
        if visual_adapter_func is not None:
            image_features, per_image_features = self.image_encoder(
                [image.type(self.dtype), compound_rep_tokens_visual, visual_adapter_func]
            )
        else:
            image_features = self.image_encoder(
                image.type(self.dtype), compound_rep_tokens_visual
            )
        return image_features, per_image_features


    def forward(self, image, idx=None):
        token_embedding, text_adapter_func, visual_adapter_func, compound_rep_tokens_text, compound_rep_tokens_visual  = self.adapter_learner()
        tokenized_prompts = self.tokenized_prompts

        if self.adapter_learner.training:
            text_features, per_text_features = self.encode_text(
                token_embedding, tokenized_prompts,compound_rep_tokens_text, text_adapter_func
            )
        else:
            if self.text_features_for_inference is None:
                self.text_features_for_inference, self.per_text_features_for_inference = self.encode_text(
                    token_embedding, tokenized_prompts, compound_rep_tokens_text, text_adapter_func
                )
            text_features = self.text_features_for_inference
            per_text_features = self.per_text_features_for_inference

        G_head_features, P_head_features = self.encode_image(image, compound_rep_tokens_visual, visual_adapter_func)

        text_features = F.normalize(text_features, dim=-1)
        per_text_features = F.normalize(per_text_features, dim=-1)

        P_head_features = F.normalize(P_head_features, dim=-1)
        G_head_features = F.normalize(G_head_features, dim=-1)

 

        logit_scale = self.logit_scale.exp()
         
        fusion_logits = logit_scale * ((1-self.Lambda)*G_head_features@ text_features.t()  +  (self.Lambda)*P_head_features@ per_text_features.t()) 
        anchor_logits = logit_scale * (G_head_features) @ text_features.t()
        # per_logits = 

        return anchor_logits, fusion_logits


class MultiModalAdapter(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.MMADAPTER.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.MMADAPTER.PREC == "fp32" or cfg.TRAINER.MMADAPTER.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()


        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        
        # for name, param in self.model.named_parameters():
        #     if "text_adapter" not in name and "visual_adapter" not in name and "shared_adapter" not in name:
        #         param.requires_grad_(False)

        names_to_update = ["text_adapter", "visual_adapter", "compound_rep_tokens_r2vproj", "compound_rep_tokens_r2tproj", "compound_rep_tokens"]

        for name, param in self.model.named_parameters():
            update = False

            for name_to_update in names_to_update:
                if name_to_update in name:
                    update = True
                    break
            param.requires_grad_(update)

        # Double check
        num_trainable_params = 0
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
                num_trainable_params += param.data.nelement()
        print(f"Parameters to be updated: {enabled}")
        print(f"Number of trainable parameters: {num_trainable_params}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("adapter_learner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MMADAPTER.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch_idx, batch, idx=-1, **kwargs):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.MMADAPTER.PREC
        if prec == "amp":
            with autocast():
                anchor_logits, fusion_logits  = self.model(image)
                running_anchor_loss = F.cross_entropy(anchor_logits, label)
                representation_fusion_loss = F.cross_entropy(fusion_logits, label)

                loss = representation_fusion_loss
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(fusion_logits, label)[0].item(),
        }

        if (batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):

        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_embedding" in state_dict:
                del state_dict["token_embedding"]
            if "tokenized_prompts" in state_dict:
                del state_dict["tokenized_prompts"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)