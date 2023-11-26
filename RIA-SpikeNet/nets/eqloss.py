import torch
import torch.nn as nn
import torch.nn.functional as F



info=[0.11719485409475996, 0.06777533730781299, 0.004706620646375902, 0.012864763100094132, 0.015061186068402886, 0.010354565422026984, 0.06291182930655789, 
0.09224976466896768, 0.03200502039535613, 0.06306871666143708, 0.025101976780671477, 0.0037652965171007216, 0.0031377470975839346, 0.005020395356134295,
 0.009099466582993411, 0.009413241292751805, 0.004235958581738312, 0.027612174458738627, 0.031377470975839344, 0.015688735487919672, 0.015688735487919672, 
 0.01490429871352369, 0.009413241292751805, 0.020866018198933165, 0.01349231251961092, 0.010354565422026984, 0.011609664261060559, 0.008001255098839033, 
 0.08440539692500784, 0.07154063382491371, 0.017885158456228428, 0.009099466582993411, 0.0031377470975839346, 0.015688735487919672, 0.015688735487919672,
  0.0032946344524631315, 0.0045497332914967055, 0.0045497332914967055, 0.004392845936617509, 0.0026670850329463445, 0.006432381550047066, 0.005177282711013492, 
  0.016630059617194853, 0.013178537809852526, 0.004706620646375902]

class EQL(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 lambda_=0.02102,
                 version="v0_5"):
        super(EQL, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.lambda_ = lambda_
        self.version = version
        self.freq_info = torch.FloatTensor(info)

        num_class_included = torch.sum(self.freq_info < self.lambda_)
        print(f"set up EQL (version {version}), {num_class_included} classes included.")

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        self.n_i, self.n_c = cls_score.size()

        self._i,self.gt_classes = label.size()
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c + 1)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target[:, :self.n_c]

        #target = expand_label(cls_score, label)
        target=label
        eql_w = 1 - self.threshold_func() * (1 - target)

        cls_loss = F.binary_cross_entropy_with_logits(cls_score, target,
                                                      reduction='none')

        cls_loss = cls_loss * eql_w

        return self.loss_weight * cls_loss

    def exclude_func(self):
        # instance-level weight
        bg_ind = self.n_c
        weight = (self.gt_classes != bg_ind).float()
        weight = weight.view(self.n_i, 1).expand(self.n_i, self.n_c)
        return weight

    def threshold_func(self):
        # class-level weight
        weight = self.pred_class_logits.new_zeros(self.n_c)
        weight[self.freq_info < self.lambda_] = 1
        weight = weight.view(1, self.n_c).expand(self.n_i, self.n_c)
        return weight
