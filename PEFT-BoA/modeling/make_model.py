import torch
import torch.nn as nn

from modeling.advanced_transformer import load_vit
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from modeling.backbones.t2t import t2t_vit_t_14, t2t_vit_t_24
from fvcore.nn import flop_count
from modeling.backbones.basic_cnn_params.flops import give_supported_ops
from timm.models.layers import trunc_normal_
from modeling.advanced_transformer.componet_module import LayerNorm
import copy


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def peft_frozen(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'trainable' not in n:
            p.requires_grad = False
    if bias == 'none':
        return


class build_transformer(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory, feat_dim):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH_T
        self.in_planes = feat_dim
        self.cv_embed_sign = cfg.MODEL.SIE_CAMERA
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.model_name = cfg.MODEL.TRANSFORMER_TYPE
        self.trans_type = cfg.MODEL.TRANSFORMER_TYPE
        self.flops_test = cfg.MODEL.FLOPS_TEST
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            self.camera_num = camera_num
        else:
            self.camera_num = 0
        # No view
        self.view_num = 0
        if False:
            pass
        elif cfg.MODEL.TRANSFORMER_TYPE == 'ViT-B-16':
            self.clip = 1
            self.sie_xishu = cfg.MODEL.SIE_COE

            vit_model = load_vit(cfg, self.model_name, cfg.INPUT.SIZE_TRAIN[0] // cfg.MODEL.STRIDE_SIZE[0],
                                 cfg.INPUT.SIZE_TRAIN[1] // cfg.MODEL.STRIDE_SIZE[1],
                                 cfg.MODEL.STRIDE_SIZE)
            print('Loading pretrained model')
            vit_model.to("cuda")
            self.base = vit_model
            if cfg.MODEL.FROZEN:
                peft_frozen(self.base)

            if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
                self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, 1, 768))
                trunc_normal_(self.cv_embed, std=.02)
                print('camera number is : {}'.format(camera_num))
            elif cfg.MODEL.SIE_CAMERA:
                self.cv_embed = nn.Parameter(torch.zeros(camera_num, 1, 768))
                trunc_normal_(self.cv_embed, std=.02)
                print('camera number is : {}'.format(camera_num))
            elif cfg.MODEL.SIE_VIEW:
                self.cv_embed = nn.Parameter(torch.zeros(view_num, 1, 768))
                trunc_normal_(self.cv_embed, std=.02)
                print('camera number is : {}'.format(view_num))

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, RGB, NI, TI, label=None, cam_label=None, view_label=None, modality=None):

        if self.cv_embed_sign:
            if self.flops_test:
                cam_label = 0
            cv_embed = self.sie_xishu * self.cv_embed[cam_label]
        else:
            cv_embed = None

        RGB, NI, TI = self.base(RGB, NI, TI, cv_embed, modality)

        RGB_global_feat = RGB[:, 0]
        NI_global_feat = NI[:, 0]
        TI_global_feat = TI[:, 0]

        RGB_feat = self.bottleneck(RGB_global_feat)
        NI_feat = self.bottleneck(NI_global_feat)
        TI_feat = self.bottleneck(TI_global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                RGB_cls_score = self.classifier(RGB_feat, label)
                NI_cls_score = self.classifier(NI_feat, label)
                TI_cls_score = self.classifier(TI_feat, label)
            else:
                RGB_cls_score = self.classifier(RGB_feat)
                NI_cls_score = self.classifier(NI_feat)
                TI_cls_score = self.classifier(TI_feat)
            return RGB, RGB_cls_score, RGB_global_feat, NI, NI_cls_score, NI_global_feat, TI, TI_cls_score, TI_global_feat
        else:
            if self.neck_feat == 'after':
                return RGB, RGB_feat, NI, NI_feat, TI, TI_feat
            else:
                return RGB, RGB_global_feat, NI, NI_global_feat, TI, TI_global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class BoA(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(BoA, self).__init__()
        if 'vit_base_patch16_224' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 768
        elif 'ViT-B-16' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 768
        self.BACKBONE = build_transformer(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)
        self.num_classes = num_classes
        self.cfg = cfg
        self.num_instance = cfg.DATALOADER.NUM_INSTANCE
        self.camera = camera_num
        self.view = view_num
        self.direct = cfg.MODEL.DIRECT
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.image_size = cfg.INPUT.SIZE_TRAIN

        self.miss_type = cfg.TEST.MISS
        self.classifier = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(3 * self.feat_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.ln_rgb = LayerNorm(self.feat_dim)
        self.ln_nir = LayerNorm(self.feat_dim)
        self.ln_tir = LayerNorm(self.feat_dim)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def flops(self, shape=(3, 256, 128)):
        if self.image_size[0] != shape[1] or self.image_size[1] != shape[2]:
            shape = (3, self.image_size[0], self.image_size[1])
            # For vehicle reid, the input shape is (3, 128, 256)
        supported_ops = give_supported_ops()
        model = copy.deepcopy(self)
        model.cuda().eval()
        input_r = torch.randn((1, *shape), device=next(model.parameters()).device)
        input_n = torch.randn((1, *shape), device=next(model.parameters()).device)
        input_t = torch.randn((1, *shape), device=next(model.parameters()).device)
        cam_label = 0
        input = {"RGB": input_r, "NI": input_n, "TI": input_t, "cam_label": cam_label}
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(
            "The out_proj here is called by the nn.MultiheadAttention, which has been calculated in th .forward(), so just ignore it.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("For the bottleneck or classifier, it is not calculated during inference, so just ignore it.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        del model, input
        return sum(Gflops.values()) * 1e9

    def forward(self, x, label=None, cam_label=None, view_label=None):
        if self.training:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']

            RGB_cash, RGB_score, RGB_global, NI_cash, NI_score, NI_global, TI_cash, TI_score, TI_global = self.BACKBONE(
                RGB, NI, TI, cam_label=cam_label, view_label=view_label)

            ori = torch.cat([self.ln_rgb(RGB_global), self.ln_nir(NI_global), self.ln_tir(TI_global)], dim=-1)

            ori_global = self.bottleneck(ori)
            ori_score = self.classifier(ori_global)

            return ori_score, ori

        else:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']

            if 'cam_label' in x:
                cam_label = x['cam_label']

            RGB_cash, RGB_global, NI_cash, NI_global, TI_cash, TI_global = self.BACKBONE(
                RGB, NI, TI, cam_label=cam_label, view_label=view_label)

            ori = torch.cat([self.ln_rgb(RGB_global), self.ln_nir(NI_global), self.ln_tir(TI_global)], dim=-1)
            return ori

        
__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    't2t_vit_t_14': t2t_vit_t_14,
    't2t_vit_t_24': t2t_vit_t_24,
}


def make_model(cfg, num_class, camera_num, view_num=0):
    model = BoA(num_class, cfg, camera_num, view_num, __factory_T_type)
    print('===========Building BoA===========')
    return model
