import torch
import config
from torch import nn as nn
from torch.nn import functional as F
from utils import get_iou, bbox_attr


class SumSquaredErrorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l_coord = 5    # Koordinat ve boyut kayıplarının ağırlığı
        self.l_noobj = 0.5  # Nesne bulunmayan bbox ların confidence kayıplarının ağırlığı

    def forward(self, p, a):    # p: tahmin edilen değerler, a: gerçek değerler
        # IOU = Modelin tahmin ettiği bbox ile gerçek bbox ın ne kadar örtüştüğünü ölçen metrik
        iou = get_iou(p, a)                     # (batch, S, S, B, B)
        max_iou = torch.max(iou, dim=-1)[0]     # (batch, S, S, B) Burada en yüksek iou değerini alıyoruz

        
        bbox_mask = bbox_attr(a, 4) > 0.0       # GT lerde confidence değeri 0 dan büyük olanları True/False olarak seçiyoruz. (batch, S, S, B) Yani orada bir nesne var mı yok mu onu kontrol ediyoruz. 1 ise nesne var, 0 ise yok
        p_template = bbox_attr(p, 4) > 0.0      # Tahmin edilen bbox larda confidence değeri 0 dan büyük olanları True/False olarak seçiyoruz. (batch, S, S, B) Yani orada bir nesne var mı yok mu onu kontrol ediyoruz. 1 ise nesne var, 0 ise yok
        obj_i = bbox_mask[..., 0:1]             # Eğer grid de bir nesne varsda ilk değeri alıyoruz (Çünkü gt tarafında nesne nilgisi ilk değerde tutuluyor)
        
        # Burada tahmin edilen en yüksek IOU değerine sahip olan bbox ın sorumlu olduğunu belirliyoruz. Yani hangi bbox nesneyi en iyi tahmin ediyorsa onu sorumlu olarak işaretliyoruz. Bu sayede loss hesaplamasında sadece sorumlu bbox lar dikkate alınıyor.
        responsible = torch.zeros_like(p_template).scatter_(       # (batch, S, S, B)
            -1,
            torch.argmax(max_iou, dim=-1, keepdim=True),                # (batch, S, S, B)
            value=1                         
        )
        obj_ij = obj_i * responsible        # Hücrede nesne varsa ve sorumlu bbox varsa 1, yoksa 0 olur. (True x True = True) (batch, S, S, B)
        noobj_ij = ~obj_ij                  # Eğer yoksa güven değeri 0 olmalıdır

        # Burada tahmin edilen ve GT bbox ların XY koordinatları arasındaki kayıpları hesaplanır.
        x_losses = mse_loss(
            obj_ij * bbox_attr(p, 0),
            obj_ij * bbox_attr(a, 0)
        )
        y_losses = mse_loss(
            obj_ij * bbox_attr(p, 1),
            obj_ij * bbox_attr(a, 1)
        )
        pos_losses = x_losses + y_losses    # Kayıplar toplanır


        # Bbox ların width, height kayıpları hesaplanır. Burada tahmin edilen width ve height değerleri sqrt alınarak normalize edilir. Bu sayede küçük bbox lar ile büyük bbox ların kayıpları arasında denge kurulur.
        p_width = bbox_attr(p, 2)
        a_width = bbox_attr(a, 2)
        width_losses = mse_loss(
            obj_ij * torch.sign(p_width) * torch.sqrt(torch.abs(p_width) + config.EPSILON),
            obj_ij * torch.sqrt(a_width)
        )
        p_height = bbox_attr(p, 3)
        a_height = bbox_attr(a, 3)
        height_losses = mse_loss(
            obj_ij * torch.sign(p_height) * torch.sqrt(torch.abs(p_height) + config.EPSILON),
            obj_ij * torch.sqrt(a_height)
        )
        dim_losses = width_losses + height_losses # Boyut kayıpları toplanır

        # Burada tahmin edilen ve GT bbox ların confidence değerleri arasındaki kayıplar hesaplanır. Eğer hücrede nesne varsa, tahmin edilen confidence değeri 1 olmalıdır. Eğer hücrede nesne yoksa, tahmin edilen confidence değeri 0 olmalıdır.
        obj_confidence_losses = mse_loss(
            obj_ij * bbox_attr(p, 4),
            obj_ij * torch.detach(max_iou)
        )

        # Sorumluluğu bulunmyan bbox ların confidence değerlerini 0 yapıyoruz
        noobj_confidence_losses = mse_loss(
            noobj_ij * bbox_attr(p, 4),
            torch.zeros_like(max_iou)
        )

        # Sınıflandırma kayıpları hesaplanır. Burada tahmin edilen ve GT sınıf değerleri arasındaki kayıplar hesaplanır. 
        class_losses = mse_loss(
            obj_i * p[..., :config.C],
            obj_i * a[..., :config.C]
        )
        
        # Bütün kayıplar toplanır ve BATCH_SIZE ile bölünerek ortlama kayıp değeri hesaplanır.
        total = self.l_coord * (pos_losses + dim_losses) \
                + obj_confidence_losses \
                + self.l_noobj * noobj_confidence_losses \
                + class_losses

        return total / config.BATCH_SIZE
        


# MSE loss fonksiyonu, tahmin edilen ve gerçek değerler arasındaki farkın karesini alarak kayıp değerini hesaplar. Burada tahmin edilen ve gerçek değerler flatten edilerek karşılaştırılır.
def mse_loss(a, b):
    flattened_a = torch.flatten(a, end_dim=-2)
    flattened_b = torch.flatten(b, end_dim=-2).expand_as(flattened_a)
    return F.mse_loss(
        flattened_a,
        flattened_b,
        reduction='sum'
    )