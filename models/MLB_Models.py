import torch.nn.functional as F
import wandb
from models.model_utils.backbone import resnet18
import einops
import copy
from models.VAVL_git.VAVL.conformer.model import Conformer
from models.model_utils.fusion_gates import *
from typing import Dict
from transformers import VivitModel, VivitConfig, ASTConfig, Wav2Vec2Model, AutoModel


#Unimodal models/encoders
class Audio_ResNet(nn.Module):
    def __init__(self, args, encs):
        super(Audio_ResNet, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.dropout if "dropout" in args else 0.1

        # self.fusion_module = ConcatFusion(output_dim=n_classes)
        # self.visual_net = resnet18(modality='visual')
        self.audio_net = resnet18(modality='audio')
        # self.vcaster = nn.Conv2d(9,3,1)
        # self.acaster = nn.Conv2d(1,3,1)
        # self.visual_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False, pretrained=False)
        # self.audio_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False, pretrained=False)
        self.aclassifier = nn.Linear(512, num_classes)


        # self.common_fc = nn.Sequential(
        #     nn.Linear(d_model, fc_inner),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(fc_inner, fc_inner),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(fc_inner, num_classes)
        # )

    def forward(self, x, **kwargs):

        # a = self.audio_net(self.acaster(x[0].unsqueeze(dim=1)))
        # pred_a = self.common_fc(a)

        audio_feat = self.audio_net(x[0].unsqueeze(dim=1))
        a = F.adaptive_avg_pool2d(audio_feat, 1)
        a = torch.flatten(a, 1)
        if "detach_enc0" in kwargs and kwargs["detach_enc0"]:
            a = a.detach()
            audio_feat = audio_feat.detach()
        if "detach_pred" in kwargs and kwargs["detach_pred"]:
            pred_a = self.aclassifier(a.detach())
        else:
            pred_a = self.aclassifier(a)


        return {"preds": {"combined": pred_a}, "features": {"combined": a}, "nonaggr_features":{"combined": audio_feat.flatten(start_dim=2)}}
class Video_ResNet(nn.Module):
    def __init__(self, args, encs):
        super(Video_ResNet, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.dropout if "dropout" in args else 0.1
        modality = args.get("modality", "visual")
        self.visual_net = resnet18(modality=modality)
        # self.vcaster = nn.Conv2d(9,3,1)
        # self.visual_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', verbose=False, weights='ResNet18_Weights.DEFAULT') # , weights='ResNet18_Weights.DEFAULT'

        # self.vclassifier = nn.Linear(512, num_classes)
        # self.vclassifier = nn.Linear(1000, num_classes)

        self.vclassifier =  nn.Sequential(
        #     nn.Linear(d_model, fc_inner),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(fc_inner, fc_inner),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, **kwargs):



        #
        # v = self.visual_net(self.vcaster(x[1].flatten(start_dim=1, end_dim=2)))
        # pred_v = self.vclassifier(v)
        # print(x[1].shape)
        v = self.visual_net(x[1])
        B = x[1].shape[0]
        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        video_feat = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(video_feat, 1)
        v = torch.flatten(v, 1)

        if "detach_enc1" in kwargs and kwargs["detach_enc1"]:
            v = v.detach()
            video_feat = video_feat.detach()

        if "detach_pred" in kwargs and kwargs["detach_pred"]:
            pred_v = self.vclassifier(v.detach())
        else:
            pred_v = self.vclassifier(v)

        return {"preds":{"combined":pred_v}, "features":{"combined":v}, "nonaggr_features":{"combined": video_feat.flatten(start_dim=2)}}
class Audio_Wav2Vec(nn.Module):
    def __init__(self, args, encs):
        super(Audio_Wav2Vec, self).__init__()

        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model

        real_model_name = "wav2vec2-large-robust"
        if self.args.get("pretrained_encoder", True):
            self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/" + real_model_name)
            self.wav2vec_model.freeze_feature_encoder()
        else:
            wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/" + real_model_name)
            config = wav2vec_model.config
            del wav2vec_model
            self.wav2vec_model = Wav2Vec2Model(config)
        if real_model_name == "wav2vec2-large-robust":
            del self.wav2vec_model.encoder.layers[12:]

        self.a_dim, self.v_dim = 1024, 1408
        self.d_v = 50
        self.hidden_2 = 512

        self.conv_1d_a = nn.Conv1d(self.a_dim, self.d_v, kernel_size=1, padding=0, bias=False)


        self.audio_net = Conformer(
                            input_dim=self.d_v,
                            encoder_dim=self.hidden_2,
                            num_encoder_layers=5)

        self.vclassifier =  nn.Sequential(
            nn.Linear(d_model, num_classes)
        )


    def forward(self, x, **kwargs):

        # print(x[2].shape)
        # self.wav2vec_model.eval()
        # with torch.no_grad():
        #     x_in = self.wav2vec_model(x[2], attention_mask=None).last_hidden_state
        if "attention_mask_audio" in x:
            x_in = self.wav2vec_model(x[2], attention_mask=x["attention_mask_audio"]).last_hidden_state
        else:
            x_in = self.wav2vec_model(x[2]).last_hidden_state

        x_in = x_in.transpose(1, 2)

        # # 1-D Convolution visual/audio features
        audio = x_in if self.a_dim == self.d_v else self.conv_1d_a(x_in)
        #
        feat_a = audio.permute(2, 0, 1)
        #
        audio_feat = self.audio_net(feat_a)
        # # print(feat_a.shape)
        #
        feat_a = nn.AdaptiveAvgPool1d(1)(audio_feat.permute(1, 2, 0)).squeeze(2)
        # feat_a = nn.AdaptiveAvgPool1d(1)(x_in).squeeze(2)
        #

        if "detach_enc0" in kwargs and kwargs["detach_enc0"]:
            feat_a = feat_a.detach()
            audio_feat = audio_feat.detach()
        if "detach_pred" in kwargs and kwargs["detach_pred"]:
            pred_a = self.vclassifier(feat_a.detach())
        else:
            pred_a = self.vclassifier(feat_a)

        # return {"preds": {"combined": pred_a}}
        return {"preds": {"combined": pred_a}, "features": {"combined": feat_a}, "nonaggr_features": {"combined": audio_feat.permute(1,2,0)}}
class Video_FacesConformer(nn.Module):
    def __init__(self, args, encs):
        super(Video_FacesConformer, self).__init__()


        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model

        self.a_dim, self.v_dim = 1024, 1408
        self.d_v = 50
        self.hidden_2 = 512

        # 1D convolutional projection layers
        self.conv_1d_v = nn.Conv1d(self.v_dim, self.d_v, kernel_size=1, padding=0, bias=False)


        self.faces_net = Conformer(
                            input_dim=self.d_v,
                            encoder_dim=self.hidden_2,
                            num_encoder_layers=5)


        self.vclassifier =  nn.Sequential(
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, **kwargs):


        x_vid = x[3].transpose(1, 2)

        # 1-D Convolution visual/audio features
        visual = x_vid if self.v_dim == self.d_v else self.conv_1d_v(x_vid)

        proj_x_v = visual.permute(2, 0, 1)
        visual_feats = self.faces_net(proj_x_v)

        feat_v = nn.AdaptiveAvgPool1d(1)(visual_feats.permute(1, 2, 0)).squeeze(2)
        # feat_a = nn.AdaptiveAvgPool1d(1)(x_in).squeeze(2)
        #
        pred_v = self.vclassifier(feat_v)



        # return {"preds": {"combined": pred_a}}
        return {"preds": {"combined": pred_v}, "features": {"combined": feat_v}, "nonaggr_features": {"combined": visual_feats.permute(1,2,0)}}

#Base models for 2 modalities
class MCR_Model(nn.Module):
    def __init__(self, args, encs):
        super(MCR_Model, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.norm_decision = args.get("norm_decision", False)



        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.batchnorm_features = args.get("batchnorm_features", False)
        self.shufflegradmulti = args.get("shufflegradmulti", False)


        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        self.count_trainingsteps = 0

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(d_model, num_classes, bias=False)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(num_classes), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(d_model, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(d_model, track_running_stats=True)

        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096, bias=False)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(4096), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(4096, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(4096, track_running_stats=True)


            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type == "nonlinear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.bias_lin = nn.Parameter(torch.zeros(fc_inner), requires_grad=True)

            if self.batchnorm_features:
                self.bn_0 = nn.BatchNorm1d(fc_inner, track_running_stats=True)
                self.bn_1 = nn.BatchNorm1d(fc_inner, track_running_stats=True)



            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
        elif self.cls_type == "film":
            self.common_fc = FiLM(512, 512, num_classes)
        elif self.cls_type == "filmv":
            self.common_fc = FiLM(512, 512, num_classes, x_film=False)
        elif self.cls_type == "gated":
            self.common_fc = GatedFusion(input_dim=512, dim=512, output_dim=num_classes)
        elif self.cls_type == "tf":
            self.common_fc = TF_Fusion(input_dim=512, dim=512, layers=2, output_dim=num_classes)
        else:
            raise ValueError("Unknown cls_type")

        if self.args.bias_infusion.get("lib", 0) > 0:
            self.fc_yz = nn.Sequential(
                nn.Linear(num_classes, d_model, bias=False),
                nn.ReLU(),
                nn.Linear(d_model, d_model*2, bias=False),
            )

    def _get_features(self, x, **kwargs):

        a = self.enc_0(x, detach_pred=not self.shufflegradmulti, **kwargs)
        v = self.enc_1(x, detach_pred=not self.shufflegradmulti, **kwargs)

        return a, v, a["preds"]["combined"], v["preds"]["combined"]

    def _forward_main(self, a, v, pred_aa, pred_vv, **kwargs):


        if self.cls_type == "linear" or self.cls_type == "highlynonlinear" or self.cls_type == "nonlinear":

            pred_a = torch.matmul(a["features"]["combined"], self.fc_0_lin.weight.T) #+ self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v["features"]["combined"], self.fc_1_lin.weight.T) #+ self.fc_0_lin.bias / 2
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_a = pred_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_v = pred_v.detach()

            if "skip_bias" in kwargs and kwargs["skip_bias"]:
                pass
            else:
                pred_v = pred_v + self.bias_lin/2
                pred_a = pred_a + self.bias_lin/2

            pred = pred_a + pred_v

            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                           "w_a": self.fc_0_lin.weight.norm(),
                           "w_v": self.fc_1_lin.weight.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

        else:
            if self.training and not kwargs.get("notwandb", False):
                wandb.log({"wf_a": pred_aa.norm(),
                           "wf_v": pred_vv.norm(),
                           "f_a": a["features"]["combined"].norm(),
                           "f_v": v["features"]["combined"].norm()
                           }, step=self.count_trainingsteps + 1)
                self.count_trainingsteps += 1

            if self.norm_decision == "standardization":
                pred_aa = (pred_aa - pred_aa.mean()) / pred_aa.std()
                pred_vv = (pred_vv - pred_vv.mean()) / pred_vv.std()
            elif self.norm_decision == "softmax":
                pred_aa = F.softmax(pred_aa, dim=1)
                pred_vv = F.softmax(pred_vv, dim=1)
            if "detach_a" in kwargs and kwargs["detach_a"]:
                pred_aa = pred_aa.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                pred_vv = pred_vv.detach()

            pred = pred_aa + pred_vv


        if self.cls_type == "film" or self.cls_type == "filmv" or self.cls_type == "gated":
            this_feat_a, this_feat_v = a["features"]["combined"], v["features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)
        elif self.cls_type == "tf":
            this_feat_a, this_feat_v = a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]
            if "detach_a" in kwargs and kwargs["detach_a"]:
                this_feat_a = this_feat_a.detach()
            if "detach_v" in kwargs and kwargs["detach_v"]:
                this_feat_v = this_feat_v.detach()
            pred = self.common_fc([this_feat_a, this_feat_v], **kwargs)

        elif self.cls_type == "nonlinear" and self.cls_type != "highlynonlinear":
            pred = self.common_fc(pred)

        return pred, pred_aa, pred_vv

    def shuffle_ids(self, label):

        batch_size = label.size(0)
        shuffle_data = []
        random_shuffling = True
        if "rand" in self.args.bias_infusion.shuffle_type:
            while len(shuffle_data) < self.args.bias_infusion.num_samples:

                if self.args.bias_infusion.shuffle:
                    shuffle_idx = torch.randperm(batch_size)
                    if "rsl" in self.args.bias_infusion.shuffle_type:
                        nonequal_label = ~(label[shuffle_idx] == label)
                        if nonequal_label.sum() <= 1:
                            continue
                        shuffle_idx = shuffle_idx[nonequal_label.cpu()]
                    elif "rsi" in self.args.bias_infusion.shuffle_type:
                        nonequal_label = ~(shuffle_idx == torch.arange(batch_size))
                        if nonequal_label.sum() <= 1:
                            continue
                        shuffle_idx = shuffle_idx[nonequal_label.cpu()]
                    else:
                        nonequal_label = torch.ones(batch_size, dtype=torch.bool)
                else:
                    nonequal_label = torch.ones(batch_size, dtype=torch.bool)
                    shuffle_idx = torch.arange(batch_size)

                if nonequal_label.sum() <= 1:
                    continue
                shuffle_data.append({"shuffle_idx": shuffle_idx, "data": nonequal_label})
        elif "samelabel" in self.args.bias_infusion.shuffle_type:
            sh_ids, data_ids = [], []
            for i, li in enumerate(label):
                for j, lj in enumerate(label):
                    if li == lj and i != j:
                        sh_ids.append(j)
                        data_ids.append(i)
            shuffle_data= [{"shuffle_idx": torch.tensor(sh_ids), "data": torch.tensor(data_ids)}]
        elif "difflabel" in self.args.bias_infusion.shuffle_type:
            sh_ids, data_ids = [], []
            for i, li in enumerate(label):
                for j, lj in enumerate(label):
                    if li != lj:
                        sh_ids.append(j)
                        data_ids.append(i)
            shuffle_data= [{"shuffle_idx": torch.tensor(sh_ids), "data": torch.tensor(data_ids)}]
        elif "alllabel" in self.args.bias_infusion.shuffle_type:
            sh_ids, data_ids = [], []
            for i, li in enumerate(label):
                for j, lj in enumerate(label):
                    if i != j:
                        sh_ids.append(j)
                        data_ids.append(i)
            shuffle_data= [{"shuffle_idx": torch.tensor(sh_ids), "data": torch.tensor(data_ids)}]
        return shuffle_data

    def shuffle_data(self, x, pred, label):
        if len(label.shape)>1:
            label = label.flatten()

        if not self.args.bias_infusion.get("training_mode", False):
            self.eval()

        a, v, pred_aa, pred_vv = self._get_features(x)

        shuffle_data = self.shuffle_ids(label)

        feat_dict = [ i for i in ["features", "nonaggr_features"] if i in a.keys() and i in v.keys() ]

        sa = {feat: {"combined": torch.concatenate([a[feat]["combined"][sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0) } for feat in feat_dict}
        sv = {feat: {"combined": torch.concatenate([v[feat]["combined"][sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0) } for feat in feat_dict}
        s_pred_aa = torch.concatenate([pred_aa[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)
        s_pred_vv = torch.concatenate([pred_vv[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)

        na = {feat: {"combined": torch.concatenate([a[feat]["combined"][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0) } for feat in feat_dict}
        nv = {feat: {"combined": torch.concatenate([v[feat]["combined"][sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0) } for feat in feat_dict}
        n_pred_aa = torch.concatenate([pred_aa[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)
        n_pred_vv = torch.concatenate([pred_vv[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        n_pred = torch.concatenate([pred[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        n_label_shuffled = torch.concatenate([label[sh_data_i["shuffle_idx"]] for sh_data_i in shuffle_data], dim=0)
        n_label = torch.concatenate([label[sh_data_i["data"]] for sh_data_i in shuffle_data], dim=0)

        if not self.args.bias_infusion.get("training_mode", False):
            self.train()

        return sa, sv, s_pred_aa, s_pred_vv, na, nv, n_pred_aa, n_pred_vv, n_pred, n_label, n_label_shuffled

    def forward(self, x, **kwargs):

        a, v, pred_aa, pred_vv = self._get_features(x, **kwargs)

        pred, pred_aa, pred_vv = self._forward_main(a, v, pred_aa, pred_vv, **kwargs)


        output = {"preds":{"combined":pred,
                            "c":pred_aa,
                            "g":pred_vv
                            },
                    "features": {"c": a["features"]["combined"],
                                "g": v["features"]["combined"]}}

        if self.training:
            if self.args.bias_infusion.get("lib", 0) > 0:
                pred_feat = self.fc_yz(pred.detach())
                combined_features = torch.cat([a["features"]["combined"], v["features"]["combined"]], dim=1)
                CMI_yz_Loss = torch.nn.MSELoss()(combined_features, pred_feat) * self.args.bias_infusion.get("lib", 0)
                output["losses"] = {"CMI_yz_Loss": CMI_yz_Loss}

            if self.args.bias_infusion.get("l", 0) != 0:

                sa, sv, s_pred_aa, s_pred_vv, na, nv, n_pred_aa, n_pred_vv, n_pred, n_label, n_label_shuffled = self.shuffle_data( x, pred, kwargs["label"])

                pred_dtv_sa, _, _ = self._forward_main(sa, nv, s_pred_aa, n_pred_vv.detach(), detach_v=True, notwandb=True, **kwargs)
                pred_dta_sa, _, _ = self._forward_main(sa, nv, s_pred_aa.detach(), n_pred_vv, detach_a=True, notwandb=True, **kwargs)
                output["preds"]["sa_detv"] = pred_dtv_sa
                output["preds"]["sa_deta"] = pred_dta_sa

                pred_dtv_sv, _, _ = self._forward_main(na, sv, n_pred_aa, s_pred_vv.detach(), detach_v=True, notwandb=True, **kwargs)
                pred_dta_sv, _, _ = self._forward_main(na, sv, n_pred_aa.detach(), s_pred_vv, detach_a=True, notwandb=True, **kwargs)
                output["preds"]["sv_detv"] = pred_dtv_sv
                output["preds"]["sv_deta"] = pred_dta_sv

                pred_sa, _, _ = self._forward_main(sa, nv, s_pred_aa.detach(), n_pred_vv.detach(), notwandb=True, **kwargs)
                pred_sv, _, _ = self._forward_main(na, sv, n_pred_aa.detach(), s_pred_vv.detach(), notwandb=True, **kwargs)

                output["preds"]["sv"] = pred_sv
                output["preds"]["sa"] = pred_sa

                output["preds"]["ncombined"] = n_pred

                output["preds"]["n_label"] = n_label
                output["preds"]["n_label_shuffled"] = n_label_shuffled

        return output
class Base_Ensemble_Model(nn.Module):
    def __init__(self, args, encs):
        super(Base_Ensemble_Model, self).__init__()

        self.args = args
        # self.shared_pred = args.shared_pred
        self.num_classes = args.num_classes
        self.norm_decision = args.get("norm_decision", False)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        if self.norm_decision == "batch_norm":
            self.norm_0 = nn.BatchNorm1d(self.num_classes , track_running_stats=False)
            self.norm_1 = nn.BatchNorm1d(self.num_classes , track_running_stats=False)
        elif self.norm_decision == "instance_norm":
            self.norm_0 = nn.InstanceNorm1d(self.num_classes , track_running_stats=False)
            self.norm_1 = nn.InstanceNorm1d(self.num_classes , track_running_stats=False)
        elif self.norm_decision == "softmax":
            self.norm_0 = nn.Softmax(dim=1)
            self.norm_1 = nn.Softmax(dim=1)


    def _get_features(self, x):
        if self.enc_0.args.get("freeze_encoder", False):
            self.enc_0.eval()
        if self.enc_1.args.get("freeze_encoder", False):
            self.enc_1.eval()

        a = self.enc_0(x)
        v = self.enc_1(x)

        return a["preds"]["combined"], v["preds"]["combined"], a["features"]["combined"], v["features"]["combined"]

    def forward(self, x, **kwargs):

        pred_a, pred_v, a, v = self._get_features(x)

        if self.norm_decision == "standardization":
            pred_a = (pred_a - pred_a.mean())/pred_a.std()
            pred_v = (pred_v - pred_v.mean())/pred_v.std()
            pred = pred_a + pred_v

        elif self.norm_decision == "batch_norm" or self.norm_decision == "instance_norm":

            pred_a = self.norm_0(pred_a)
            pred_v = self.norm_1(pred_v)
            pred = pred_a + pred_v

        elif self.norm_decision == "softmax":

            pred_a = self.norm_0(pred_a)
            pred_v = self.norm_1(pred_v)
            pred = torch.nn.functional.softmax(pred_a, dim=1) + torch.nn.functional.softmax(pred_v, dim=1)
        else:
            pred = pred_a + pred_v

        if a.shape != v.shape:
            return {"preds":{"combined":pred, "c":pred_a, "g":pred_v}, "features": {"c": a, "g": v}}
        return {"preds":{"combined":pred, "c":pred_a, "g":pred_v}, "features": {"c": a, "g": v, "combined": (a + v)/2}}
class Base_Model(nn.Module):
    def __init__(self, args, encs):
        super(Base_Model, self).__init__()

        self.args = args
        # self.shared_pred = args.shared_pred
        self.cls_type = args.cls_type

        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.cls_type = args.get("cls_type", "linear")

        self.mmcosine = args.get("mmcosine", False)
        self.mmcosine_scaling = args.get("mmcosine_scaling", 10)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        if self.cls_type == "linear" or self.cls_type == "linear_stopgrad" or self.cls_type =="linear_ogm" or self.cls_type == "linear_ogm_multi":
            self.fc_0_lin = nn.Linear(d_model, num_classes)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
        elif self.cls_type == "dec":
            pass

        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type == "nonlinear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )

        elif self.cls_type == "film":
            self.common_fc = FiLM(512, 512, num_classes)
        elif self.cls_type == "gated":
            self.common_fc = GatedFusion(input_dim=512, dim=512, output_dim=num_classes)
        elif self.cls_type == "tf":
            self.common_fc = TF_Fusion(input_dim=512, dim=512, layers=2, output_dim=num_classes)
        else:
            raise ValueError("Unknown cls_type")
    def _get_features(self, x, detach_pred=False):

        a = self.enc_0(x, detach_pred=detach_pred)
        v = self.enc_1(x, detach_pred=detach_pred)

        return a, v, a["preds"]["combined"], v["preds"]["combined"]

    def forward(self, x, **kwargs):

        if self.cls_type == "linear_stopgrad":
            a, v, pred_aa, pred_vv = self._get_features(x, detach_pred=True)
        else:
            a, v, pred_aa, pred_vv = self._get_features(x)

        if self.mmcosine:
            pred_a = torch.mm(F.normalize(a["features"]["combined"], dim=1),F.normalize(torch.transpose(self.fc_0_lin.weight, 0, 1), dim=0))  # w[n_classes,feature_dim*2]->W[feature_dim, n_classes], norm at dim 0.
            pred_v = torch.mm(F.normalize(v["features"]["combined"], dim=1), F.normalize(torch.transpose(self.fc_1_lin.weight, 0, 1), dim=0))
            pred_a = pred_a * self.mmcosine_scaling
            pred_v = pred_v * self.mmcosine_scaling
            pred = pred_a + pred_v
        elif self.cls_type == "linear_ogm":
            pred_a = torch.matmul(a["features"]["combined"], self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v["features"]["combined"], self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 2
            pred = pred_a + pred_v
            pred_aa = pred_a.detach()
            pred_vv = pred_v.detach()
        elif self.cls_type == "linear_ogm_multi":
            pred_a = torch.matmul(a["features"]["combined"], self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v["features"]["combined"], self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 2
            pred = pred_a + pred_v
            pred_aa = pred_a
            pred_vv = pred_v
        elif self.cls_type == "linear" or self.cls_type == "highlynonlinear" or self.cls_type == "nonlinear" or "linear_stopgrad":

            pred_a = torch.matmul(a["features"]["combined"], self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v["features"]["combined"], self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 2

            pred = pred_a + pred_v
        else:
            pred = pred_aa + pred_vv

        if self.cls_type == "film" or self.cls_type == "gated":
            pred = self.common_fc([a["features"]["combined"], v["features"]["combined"]])
        elif self.cls_type == "tf":
            pred = self.common_fc([a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]])
        elif self.cls_type == "nonlinear" and self.cls_type != "highlynonlinear":
            pred = self.common_fc(pred)
        if (self.args.bias_infusion.method == "OGM" or self.args.bias_infusion.method == "OGM_GE" or self.args.bias_infusion.method == "MSLR") and self.cls_type!="dec":
            pred_aa = pred_a
            pred_vv = pred_v

        return {"preds":{"combined":pred,
                         "c":pred_aa,
                         "g":pred_vv
                         },
                "features": {"c": a["features"]["combined"],
                             "g": v["features"]["combined"]}}
class AGM_Model(nn.Module):
    def __init__(self, args, encs):
        super(AGM_Model, self).__init__()

        self.args = args
        # self.shared_pred = args.shared_pred
        self.cls_type = args.cls_type

        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.mmcosine = args.get("mmcosine", False)
        self.mmcosine_scaling = args.get("mmcosine_scaling", 10)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(d_model, num_classes)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
        elif self.cls_type == "dec":
            pass
        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type == "nonlinear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )

        elif self.cls_type == "film":
            self.common_fc = FiLM(d_model, 512, num_classes)
        elif self.cls_type == "gated":
            self.common_fc = GatedFusion(input_dim=d_model, dim=512, output_dim=num_classes)
        elif self.cls_type == "tf":
            self.common_fc = TF_Fusion(input_dim=d_model, dim=512, layers=2, output_dim=num_classes)
        else:
            raise ValueError("Unknown cls_type")

        self.m_v_o = Modality_out()
        self.m_a_o = Modality_out()

        self.scale_a = 1.0
        self.scale_v = 1.0

        self.m_a_o.register_full_backward_hook(self.hooka)
        self.m_v_o.register_full_backward_hook(self.hookv)

    def hooka(self, m, ginp, gout):
        gnew = ginp[0].clone()
        return gnew * self.scale_a,

    def hookv(self, m, ginp, gout):
        gnew = ginp[0].clone()
        return gnew * self.scale_v,

    def update_scale(self, coeff_a, coeff_v):
        self.scale_a = coeff_a
        self.scale_v = coeff_v

    def _get_features(self, x):

        a = self.enc_0(x)
        v = self.enc_1(x)

        return a, v, a["preds"]["combined"], v["preds"]["combined"]

    def _get_preds_padded(self, x, feat_a, feat_v, pred_aa, pred_vv, pad_audio = False, pad_visual = False):

        data = copy.deepcopy(x)
        if pad_audio:
            if 0 in data:
                data[0] = torch.zeros_like(data[0], device=x[0].device)
            elif 2 in data:
                data[2] = torch.zeros_like(data[2], device=x[2].device)
            a = self.enc_0(data)
            if self.cls_type == "dec":
                pred = a["preds"]["combined"] + pred_vv
            else:
                pred = self._forward_main(a, feat_v)

        if pad_visual:
            if 1 in data:
                data[1] = torch.zeros_like(data[1], device=x[1].device)
            elif 3 in data:
                data[3] = torch.zeros_like(data[3], device=x[3].device)
            v = self.enc_1(data)
            if self.cls_type == "dec":
                pred = pred_aa + v["preds"]["combined"]
            else:
                pred = self._forward_main(feat_a, v)

        return pred

    def _forward_main(self, a, v):

        if self.mmcosine:
            pred_a = torch.mm(F.normalize(a["features"]["combined"], dim=1),F.normalize(torch.transpose(self.fc_0_lin.weight, 0, 1), dim=0))  # w[n_classes,feature_dim*2]->W[feature_dim, n_classes], norm at dim 0.
            pred_v = torch.mm(F.normalize(v["features"]["combined"], dim=1), F.normalize(torch.transpose(self.fc_1_lin.weight, 0, 1), dim=0))
            pred_a = pred_a * self.mmcosine_scaling
            pred_v = pred_v * self.mmcosine_scaling
            pred = pred_a + pred_v
        elif self.cls_type == "linear" or self.cls_type == "highlynonlinear" or self.cls_type == "nonlinear":
            pred_a = torch.matmul(a["features"]["combined"], self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 2
            pred_v = torch.matmul(v["features"]["combined"], self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 2
            pred = pred_a + pred_v

        if self.cls_type == "film" or self.cls_type == "gated":
            pred = self.common_fc([a["features"]["combined"], v["features"]["combined"]])
        elif self.cls_type == "tf":
            pred = self.common_fc([a["nonaggr_features"]["combined"], v["nonaggr_features"]["combined"]])

        # if self.cls_type == "film" or self.cls_type == "gated" or self.cls_type == "tf":
        #     pred = self.common_fc([a, v])
        #     return pred
        elif self.cls_type != "linear":
            pred = self.common_fc(pred)

        return pred


    def forward(self, x, **kwargs):

        a, v, pred_aa, pred_vv = self._get_features(x)

        training_mode = True if self.training else False
        self.eval()
        pred_za = self._get_preds_padded(x, feat_a=a, feat_v=v, pred_aa=pred_aa, pred_vv=pred_vv,pad_audio=True, pad_visual=False)
        pred_zv = self._get_preds_padded(x, feat_a=a, feat_v=v, pred_aa=pred_aa, pred_vv=pred_vv,pad_audio=False, pad_visual=True)
        if training_mode:
            self.train()

        if self.cls_type == "dec":
            pred = pred_aa + pred_vv
        else:
            pred = self._forward_main(a, v)

        pred_a = self.m_a_o(0.5*(pred - pred_za + pred_zv))
        pred_v = self.m_v_o(0.5*(pred - pred_zv + pred_za))


        return {"preds":{"combined":pred_a + pred_v,
                         "both": pred,
                         "c":pred_a,
                         "g":pred_v
                         },
                "features": {"c": a["features"]["combined"],
                             "g": v["features"]["combined"]}}
class Modality_out(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x
class MLA_Model(nn.Module):
    def __init__(self, args, encs):
        super(MLA_Model, self).__init__()

        self.args = args
        # self.shared_pred = args.shared_pred
        self.num_classes = args.num_classes
        self.norm_decision = args.get("norm_decision", False)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        self.fc_out = nn.Linear(self.enc_0.args["d_model"], self.num_classes)

    def calculate_entropy(self, output):
        probabilities = F.softmax(output, dim=0)
        # probabilities = F.softmax(output, dim=1)
        log_probabilities = torch.log(probabilities)
        entropy = -torch.sum(probabilities * log_probabilities)
        return entropy

    def calculate_gating_weights(self, encoder_output_1, encoder_output_2):

        entropy_1 = self.calculate_entropy(encoder_output_1)
        entropy_2 = self.calculate_entropy(encoder_output_2)

        max_entropy = max(entropy_1, entropy_2)

        gating_weight_1 = torch.exp(max_entropy - entropy_1)
        gating_weight_2 = torch.exp(max_entropy - entropy_2)

        sum_weights = gating_weight_1 + gating_weight_2

        gating_weight_1 /= sum_weights
        gating_weight_2 /= sum_weights

        return gating_weight_1, gating_weight_2

    def _get_features(self, x):
        if self.enc_0.args.get("freeze_encoder", False):
            self.enc_0.eval()
        if self.enc_1.args.get("freeze_encoder", False):
            self.enc_1.eval()

        a = self.enc_0(x)
        v = self.enc_1(x)

        return a["preds"]["combined"], v["preds"]["combined"], a["features"]["combined"], v["features"]["combined"]

    def forward(self, x, **kwargs):

        _, _, a, v = self._get_features(x)

        pred_a = self.fc_out(a)
        pred_v = self.fc_out(v)

        if self.args.bias_infusion.dynamic:
            a_conf, v_conf = self.calculate_gating_weights(pred_a, pred_v)
            pred = (pred_a * a_conf + pred_v * v_conf)
        else:
            pred = self.args.bias_infusion.alpha * pred_a + (1 - self.args.bias_infusion.alpha) * pred_v

        if a.shape != v.shape:
            return {"preds":{"combined":pred, "c":pred_a, "g":pred_v}, "features": {"c": a, "g": v}}
        return {"preds":{"combined":pred, "c":pred_a, "g":pred_v}, "features": {"c": a, "g": v, "combined": (a + v)/2}}

class Base_3D_Model(nn.Module):
    def __init__(self, args, encs):
        super(Base_3D_Model, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.precondition = args.get("precondition", False)
        self.norm_decision = args.get("norm_decision", False)

        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.mmcosine = args.get("mmcosine", False)
        self.mmcosine_scaling = args.get("mmcosine_scaling", 10)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]
        self.enc_2 = encs[2]

        self.dropout = nn.Dropout(0.3)

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(d_model, num_classes)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
            self.fc_2_lin = nn.Linear(d_model, num_classes, bias=False)

            if self.precondition:
                self.inst_norm = nn.InstanceNorm1d(num_classes, track_running_stats=False)
                self.precond_norm = nn.BatchNorm1d(num_classes, track_running_stats=False)

        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)
            self.fc_2_lin = nn.Linear(d_model, 4096, bias=False)

            if self.precondition:
                self.inst_norm = nn.InstanceNorm1d(d_model, track_running_stats=False)
                self.precond_norm = nn.BatchNorm1d(d_model, track_running_stats=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type == "nonlinear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.fc_2_lin =  nn.Linear(d_model, fc_inner, bias=False)

            if self.precondition:
                self.inst_norm = nn.InstanceNorm1d(d_model, track_running_stats=False)
                self.precond_norm = nn.BatchNorm1d(d_model, track_running_stats=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )
        elif self.cls_type == "tf":
            self.common_fc = TF_Fusion(input_dim=40, dim=40, layers=2, output_dim=num_classes)
        # else:
        #     raise ValueError("Unknown cls_type")


    def forward(self, x, **kwargs):


        a = self.enc_0(x)
        v = self.enc_1(x)
        z = self.enc_2(x)
        feat_z  = z["features"]["combined"]
        feat_a = a["features"]["combined"]
        feat_v = v["features"]["combined"]
        pred_aa = a["preds"]["combined"]
        pred_vv = v["preds"]["combined"]
        pred_zz = z["preds"]["combined"]

        if self.mmcosine:
            pred_a = torch.mm(F.normalize(feat_a, dim=1),F.normalize(torch.transpose(self.fc_0_lin.weight, 0, 1), dim=0))  # w[n_classes,feature_dim*2]->W[feature_dim, n_classes], norm at dim 0.
            pred_v = torch.mm(F.normalize(feat_v, dim=1), F.normalize(torch.transpose(self.fc_1_lin.weight, 0, 1), dim=0))
            pred_z = torch.mm(F.normalize(feat_z, dim=1), F.normalize(torch.transpose(self.fc_2_lin.weight, 0, 1), dim=0))
            pred_a = pred_a * self.mmcosine_scaling
            pred_v = pred_v * self.mmcosine_scaling
            pred_z = pred_z * self.mmcosine_scaling

            pred = pred_a + pred_v + pred_z

        elif self.cls_type == "linear" or self.cls_type == "highlynonlinear" or self.cls_type == "nonlinear":

            pred_a = torch.matmul(feat_a, self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 3
            pred_v = torch.matmul(feat_v, self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 3
            pred_z = torch.matmul(feat_z, self.fc_2_lin.weight.T) + self.fc_0_lin.bias / 3
            pred = pred_a + pred_v + pred_z
        else:
            if self.norm_decision == "standardization":
                pred_aa = (pred_aa - pred_aa.mean()) / pred_aa.std()
                pred_vv = (pred_vv - pred_vv.mean()) / pred_vv.std()
                pred_zz = (pred_zz - pred_zz.mean()) / pred_zz.std()
            elif self.norm_decision == "softmax":
                pred_aa = F.softmax(pred_aa, dim=1)
                pred_vv = F.softmax(pred_vv, dim=1)
                pred_zz = F.softmax(pred_zz, dim=1)

            pred = pred_aa + pred_vv + pred_zz

        if self.cls_type == "film" or self.cls_type == "gated":
            pred = self.common_fc([a["features"]["combined"], v["features"]["combined"], z["features"]["combined"]])
        elif self.cls_type == "tf":
            tf_input = [a["nonaggr_features"]["combined"].permute(1,2,0),
                        v["nonaggr_features"]["combined"].permute(1,2,0),
                        z["nonaggr_features"]["combined"].permute(1,2,0)]
            pred = self.common_fc(tf_input)
        elif self.cls_type == "nonlinear" and self.cls_type != "highlynonlinear":
            pred = self.common_fc(pred)
        bias_method = self.args.get("bias_infusion", {"method": False}).get("method", False)
        if (bias_method == "OGM" or bias_method == "OGM_GE" or bias_method == "MSLR") and self.cls_type!="dec":
            pred_aa = pred_a
            pred_vv = pred_v
            pred_zz = pred_z


        output = {"preds":{
            "combined":pred,
            "c":pred_aa,
            "g":pred_vv,
            "f":pred_zz},
                  "features":{"c":feat_a,
                              "g":feat_v,
                              "f":feat_z}}
        return output
class AGM_3D_Model(nn.Module):
    def __init__(self, args, encs):
        super(AGM_3D_Model, self).__init__()

        self.args = args
        self.cls_type = args.cls_type
        self.precondition = args.get("precondition", False)
        self.norm_decision = args.get("norm_decision", False)

        num_classes = args.num_classes
        d_model = args.d_model
        fc_inner = args.fc_inner
        dropout = args.get("dropout", 0.1)

        self.mmcosine = args.get("mmcosine", False)
        self.mmcosine_scaling = args.get("mmcosine_scaling", 10)

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]
        self.enc_2 = encs[2]

        self.dropout = nn.Dropout(0.3)

        if self.cls_type == "linear":
            self.fc_0_lin = nn.Linear(d_model, num_classes)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
            self.fc_2_lin = nn.Linear(d_model, num_classes, bias=False)

            if self.precondition:
                self.inst_norm = nn.InstanceNorm1d(num_classes, track_running_stats=False)
                self.precond_norm = nn.BatchNorm1d(num_classes, track_running_stats=False)

        elif self.cls_type == "highlynonlinear":
            self.fc_0_lin = nn.Linear(d_model, 4096)
            self.fc_1_lin = nn.Linear(d_model, 4096, bias=False)
            self.fc_2_lin = nn.Linear(d_model, 4096, bias=False)

            if self.precondition:
                self.inst_norm = nn.InstanceNorm1d(d_model, track_running_stats=False)
                self.precond_norm = nn.BatchNorm1d(d_model, track_running_stats=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        elif self.cls_type == "nonlinear":
            self.fc_0_lin = nn.Linear(d_model, fc_inner)
            self.fc_1_lin = nn.Linear(d_model, fc_inner, bias=False)
            self.fc_2_lin =  nn.Linear(d_model, fc_inner, bias=False)

            if self.precondition:
                self.inst_norm = nn.InstanceNorm1d(d_model, track_running_stats=False)
                self.precond_norm = nn.BatchNorm1d(d_model, track_running_stats=False)

            self.common_fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, fc_inner),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_inner, num_classes)
            )

        elif self.cls_type == "tf":

            self.fc_0_lin = nn.Linear(d_model, num_classes)
            self.fc_1_lin = nn.Linear(d_model, num_classes, bias=False)
            self.fc_2_lin = nn.Linear(d_model, num_classes, bias=False)

            self.common_fc = TF_Fusion(input_dim=40, dim=40, layers=2, output_dim=num_classes)
        else:
            raise ValueError("Unknown cls_type")
        self.m_v_o = Modality_out()
        self.m_f_o = Modality_out()
        self.m_l_o = Modality_out()

        self.m_f = Modality_Text()
        self.m_l = Modality_Audio()
        self.m_v = Modality_Visual()
        self.m_v_o = Modality_out()
        self.m_f_o = Modality_out()
        self.m_l_o = Modality_out()

        self.scale_f = 1.0
        self.scale_v = 1.0
        self.scale_l = 1.0

        self.m_f_o.register_full_backward_hook(self.hookf)
        self.m_v_o.register_full_backward_hook(self.hookv)
        self.m_l_o.register_full_backward_hook(self.hookl)

        # if cfg.CHECKPOINT_PATH:
        #     print("We are loading from {}".format(cfg.CHECKPOINT_PATH))
        #     self.load_state_dict(torch.load(cfg.CHECKPOINT_PATH, map_location="cpu"))

    def hookl(self, m, ginp, gout):
        gnew = ginp[0].clone()
        return gnew * self.scale_l,

    def hookv(self, m, ginp, gout):
        gnew = ginp[0].clone()
        return gnew * self.scale_v,

    def hookf(self, m, ginp, gout):
        gnew = ginp[0].clone()
        return gnew * self.scale_f,

    def update_scale(self,coeff_c,coeff_g,coeff_f):
        self.scale_v = coeff_c
        self.scale_l = coeff_g
        self.scale_f = coeff_f


    def make_zero_batch(self, batch: Dict[str, torch.Tensor]):
        zero_input = {}
        for key in batch:
            zero_input[key] = torch.zeros_like(batch[key])
        return zero_input

    def classifier(self, features, noaggr_features, pred_aa, pred_vv, pred_zz, return_all=False, **kwargs):
        feat_a = features["c"]
        feat_v = features["g"]
        feat_z = features["flow"]
        if self.cls_type == "linear" or self.cls_type == "highlynonlinear" or self.cls_type == "nonlinear":

            pred_a = torch.matmul(feat_a, self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 3
            pred_v = torch.matmul(feat_v, self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 3
            pred_z = torch.matmul(feat_z, self.fc_2_lin.weight.T) + self.fc_0_lin.bias / 3
            preds = pred_a + pred_v + pred_z

        elif self.cls_type == "tf":
            tf_input = [noaggr_features["c"].permute(1, 2, 0),
                        noaggr_features["g"].permute(1, 2, 0),
                        noaggr_features["flow"].permute(1, 2, 0)]

            pred_aa = torch.matmul(feat_a, self.fc_0_lin.weight.T) + self.fc_0_lin.bias / 3
            pred_vv = torch.matmul(feat_v, self.fc_1_lin.weight.T) + self.fc_0_lin.bias / 3
            pred_zz = torch.matmul(feat_z, self.fc_2_lin.weight.T) + self.fc_0_lin.bias / 3

            preds = self.common_fc(tf_input)

        elif self.cls_type == "nonlinear" and self.cls_type != "highlynonlinear":
            preds = self.common_fc(torch.concatenate([feat_a, feat_v, feat_z], dim=1))
        else:
            if self.norm_decision == "standardization":
                pred_aa = (pred_aa - pred_aa.mean()) / pred_aa.std()
                pred_vv = (pred_vv - pred_vv.mean()) / pred_vv.std()
                pred_zz = (pred_zz - pred_zz.mean()) / pred_zz.std()
            elif self.norm_decision == "softmax":
                pred_aa = F.softmax(pred_aa, dim=1)
                pred_vv = F.softmax(pred_vv, dim=1)
                pred_zz = F.softmax(pred_zz, dim=1)

            preds = pred_aa + pred_vv + pred_zz
        if return_all:
            return preds, pred_aa, pred_vv, pred_zz
        return preds

    def forward(self, x, **kwargs):


        a = self.enc_0(x)
        v = self.enc_1(x)
        z = self.enc_2(x)
        feat_z  = z["features"]["combined"]
        feat_a = a["features"]["combined"]
        feat_v = v["features"]["combined"]
        nonaggr_feat_z  = z["nonaggr_features"]["combined"]
        nonaggr_feat_a = a["nonaggr_features"]["combined"]
        nonaggr_feat_v = v["nonaggr_features"]["combined"]
        pred_aa = a["preds"]["combined"]
        pred_vv = v["preds"]["combined"]
        pred_zz = z["preds"]["combined"]

        features = {"c": feat_a, "g": feat_v, "flow": feat_z}
        nonaggr_features = {"c": nonaggr_feat_a, "g": nonaggr_feat_v, "flow": nonaggr_feat_z}

        preds, pred_aa, pred_vv, pred_zz = self.classifier(features, nonaggr_features, pred_aa, pred_vv, pred_zz, return_all = True)


        train_flag = self.training == 'train'

        self.eval()
        with torch.no_grad():
            zero_input = self.make_zero_batch(x)

            v = self.enc_0(zero_input, return_features=True)
            l = self.enc_1(zero_input, return_features=True)
            f = self.enc_2(zero_input, return_features=True)
            video_zero_features = v["features"]["combined"]
            layout_zero_features = l["features"]["combined"]
            flow_zero_features = f["features"]["combined"]
            video_zero_noaggr_features = v["nonaggr_features"]["combined"]
            layout_zero_noaggr_features = l["nonaggr_features"]["combined"]
            flow_zero_noaggr_features = f["nonaggr_features"]["combined"]

            features = {"c": video_zero_features, "g": feat_v, "flow": feat_z}
            nonaggr_features = {"c": video_zero_noaggr_features, "g": nonaggr_feat_v, "flow": nonaggr_feat_z}
            preds_zv = self.classifier(features, nonaggr_features, pred_aa, pred_vv, pred_zz)

            features = {"c": feat_a, "g": layout_zero_features, "flow": feat_z}
            nonaggr_features = {"c": nonaggr_feat_a, "g": layout_zero_noaggr_features, "flow": nonaggr_feat_z}
            preds_zl = self.classifier(features, nonaggr_features, pred_aa, pred_vv, pred_zz)

            features = {"c": feat_a, "g": feat_v, "flow": flow_zero_features}
            nonaggr_features = {"c": nonaggr_feat_a, "g": nonaggr_feat_v, "flow": flow_zero_noaggr_features}
            preds_zf = self.classifier(features, nonaggr_features, pred_aa, pred_vv, pred_zz)

            features = {"c": video_zero_features, "g": layout_zero_features, "flow": feat_z}
            nonaggr_features = {"c": video_zero_noaggr_features, "g": layout_zero_noaggr_features, "flow": nonaggr_feat_z}
            preds_zvl = self.classifier(features, nonaggr_features, pred_aa, pred_vv, pred_zz)

            features = {"c": video_zero_features, "g": feat_v, "flow": flow_zero_features}
            nonaggr_features = {"c": video_zero_noaggr_features, "g": nonaggr_feat_v, "flow": flow_zero_noaggr_features}
            preds_zvf = self.classifier(features, nonaggr_features, pred_aa, pred_vv, pred_zz)

            features = {"c": feat_a, "g": layout_zero_features, "flow": flow_zero_features}
            nonaggr_features = {"c": nonaggr_feat_a, "g": layout_zero_noaggr_features, "flow": flow_zero_noaggr_features}
            preds_zlf = self.classifier(features, nonaggr_features,pred_aa, pred_vv, pred_zz)

            features = {"c": video_zero_features, "g": layout_zero_features, "flow": feat_z}
            nonaggr_features = {"c": video_zero_noaggr_features, "g": layout_zero_noaggr_features, "flow": nonaggr_feat_z}
            preds_zvlf = self.classifier(features, nonaggr_features, pred_aa, pred_vv, pred_zz)


        if train_flag: self.train()
        m_v_out = self.m_v_o(self.m_v(preds,
                                      preds_zv, preds_zl, preds_zf,
                                      preds_zvl, preds_zvf, preds_zlf,
                                      preds_zvlf))
        m_l_out = self.m_l_o(self.m_l(preds,
                                      preds_zv, preds_zl, preds_zf,
                                      preds_zvl, preds_zvf, preds_zlf,
                                      preds_zvlf))
        m_f_out = self.m_f_o(self.m_f(preds,
                                      preds_zv, preds_zl, preds_zf,
                                      preds_zvl, preds_zvf, preds_zlf,
                                      preds_zvlf))

        # individual marginal contribution (contain zero padding)
        m_l_mc = m_l_out - preds_zvlf / 3
        m_v_mc = m_v_out - preds_zvlf / 3
        m_f_mc = m_f_out - preds_zvlf / 3
        pred = {}
        pred.update({"both": preds})
        pred.update({"combined": m_v_out + m_l_out + m_f_out})
        pred.update({"c_mc": m_v_mc})
        pred.update({"g_mc": m_l_mc})
        pred.update({"f_mc": m_f_mc})
        pred.update({"c": m_v_out})
        pred.update({"g": m_l_out})
        pred.update({"f": m_f_out})

        output = {"preds":pred,
                  "features":{"c":feat_a,
                              "g":feat_v,
                              "f":feat_z}}
        return output

class Modality_Text(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,total_out,
                pad_visual_out,pad_audio_out,pad_text_out,
                pad_visual_audio_out,pad_visual_text_out,pad_audio_text_out,
                zero_padding_out):
        return (total_out-pad_text_out+pad_visual_audio_out)/3 + (pad_visual_out - pad_audio_text_out+pad_audio_out-pad_visual_text_out)/6
class Modality_Audio(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,total_out,
                pad_visual_out,pad_audio_out,pad_text_out,
                pad_visual_audio_out,pad_visual_text_out,pad_audio_text_out,
                zero_padding_out):
        return (total_out-pad_audio_out+pad_visual_text_out) / 3 + (pad_visual_out - pad_audio_text_out + pad_text_out - pad_visual_audio_out) / 6
class Modality_Visual(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,total_out,
                pad_visual_out,pad_audio_out,pad_text_out,
                pad_visual_audio_out,pad_visual_text_out,pad_audio_text_out,
                zero_padding_out):
        return (total_out-pad_visual_out+pad_audio_text_out)/3 + (pad_audio_out-pad_visual_text_out + pad_text_out - pad_visual_audio_out)/6
class TF_Fusion(nn.Module):
    def __init__(self, input_dim, dim, layers, output_dim):
        super(TF_Fusion, self).__init__()
        self.common_net = Conformer(
                            input_dim=input_dim,
                            encoder_dim=dim,
                            num_encoder_layers=layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim), requires_grad=True)
        self.mod_0_token = nn.Parameter(torch.randn(1, 1, input_dim), requires_grad=True)
        self.mod_1_token = nn.Parameter(torch.randn(1, 1, input_dim), requires_grad=True)
        self.mod_2_token = nn.Parameter(torch.randn(1, 1, input_dim), requires_grad=True)

        self.common_fc = nn.Linear(dim, output_dim)


    def forward(self, x, **kwargs):
        x_0 = x[0].permute(0,2,1)
        x_1 = x[1].permute(0,2,1)

        x_0 = self.mod_0_token.repeat(x_0.shape[0], x_0.shape[1], 1) + x_0
        x_1 = self.mod_1_token.repeat(x_1.shape[0], x_1.shape[1], 1) + x_1
        xlist = [x_0, x_1]
        if len(x)>2:
            x_2 = x[2].permute(0,2,1)
            x_2 = self.mod_2_token.repeat(x_2.shape[0], x_2.shape[1], 1) + x_2
            xlist.append(x_2)
        if "detach_a" in kwargs and kwargs["detach_a"]:
            xlist[0] = xlist[0].detach()
        if "detach_v" in kwargs and kwargs["detach_v"]:
            xlist[1] = xlist[1].detach()

        feat_mm = torch.concatenate([xi for xi in xlist], dim=1)
        feat_mm = torch.concatenate([self.cls_token.repeat(feat_mm.shape[0], 1, 1), feat_mm], dim=1)
        feat_mm = self.common_net(feat_mm)
        aggr_feat_mm = feat_mm[:,0]

        pred = self.common_fc(aggr_feat_mm)
        if kwargs.get("return_all", False):
            return pred, aggr_feat_mm, feat_mm
        else:
            return pred


