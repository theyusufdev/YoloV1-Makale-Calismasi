import random
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.datasets.voc import VOCDetection
import torch
import config
import utils

# Bu sınıfın amacı,  dışarıdan gelen ham nesne kutularını ve sınıf etiketlerini alıp modelin eğilim sırasında doğrudan kullanabileceği 7 * 7 * 30 boyutundaki düzenli matrise dönüştürmektir.
class YoloPascalVOCDataset(Dataset):

    def __init__(self, data_path, set_type, is_augment: bool = True):
        self.dataset = VOCDetection(
            root=data_path,
            year='2007',
            image_set=('train' if set_type == 'train' else 'val'),
            download=False,
        )

        self.augment = is_augment
        self.classes = {
            name: int(index)
            for index, name in utils.load_class_dict().items()
        }

    # __getitem__ metodu, veri kümesindeki bir örneği alır ve onu modelin eğitimi için uygun formata dönüştürür. Bu, görüntüyü yeniden boyutlandırmayı, veri artırmayı ve sınıf etiketlerini ve nesne kutularını uygun şekilde düzenlemeyi içerir.
    def __getitem__(self, i):
        data, label = self.dataset[i]  # data = resim, label = resimdeki nesnelerin sınıf etiketleri ve koordinatları
        data = F.resize(img=data, size=config.IMAGE_SIZE)       # Görseli makelede belirtilen (448,448) boyutuna yeniden boyutlandırır

        original_data = F.to_tensor(data)
        x_shift = int((0.2 * random.random() - 0.1) * config.IMAGE_SIZE[0]) # Görselin makalede belirtildiği gibi rastgele kaydırılması için x ekseninde kaydırma miktarını hesaplar
        y_shift = int((0.2 * random.random() - 0.1) * config.IMAGE_SIZE[1]) # Görselin makalede belirtildiği gibi rastgele kaydırılması için y ekseninde kaydırma miktarını hesaplar
        scale = 1 + 0.2 * random.random()                                   # Görselin ölçeklendirilmesi için %20 oranona varan rastgele bir ölçeklendirme değerini hesaplar

        if self.augment:
            data = F.affine(img=data, translate=(x_shift, y_shift), scale=scale, shear=0.0, angle=0.0)    # Görseli rastgele kaydırır ve ölçeklendirir
            data = F.adjust_brightness(img=data, brightness_factor=0.5)                                   # Görselin parlaklığını rastgele değiştirir
            data = F.adjust_saturation(img=data, saturation_factor=0.5)                                   # Görselin doygunluğunu rastgele değiştirir
            data = F.to_tensor(pic=data)                                                                          # Görseli tensöre dönüştürür 
            data = F.normalize(tensor=data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])                     # Görselin piksel değerlerini [0, 1] aralığına normalleştirir

        grid_size_x = data.size(dim=2) / config.S  # Görselin genişliğini S (7) ile bölerek her bir grid hücresinin genişliğini hesaplar (dim=2 -> width)
        grid_size_y = data.size(dim=1) / config.S  # Görselin yüksekliğini S (7) ile bölerek her bir grid hücresinin yüksekliğini hesaplar (dim=1 -> height)

        # Bounding Box, sınıf ve konum bilgilerini içeren 7x7x30 boyutunda bir tensör oluşturur. Bu tensör, modelin eğitimi sırasında kullanılacak olan ground truth verilerini temsil eder.
        boxes = {}                                                          # Bbox ların hangi grid hücresine ait olduğunu takip etmek için bir sözlük
        class_names = {}                                                    # Her bir grid hücresinin hangi sınıfta olduğunu takip etmek için bir sözlük
        depth = 5 * config.B + config.C                                     # Her bbox için 5 özellik (x, y, w, h, confidence) ve sınıf sayısı (C) ile toplam derinliğini hesapalr
        ground_truth = torch.zeros((config.S, config.S, depth))             # Her bbox için (7,7,depth) boyutunda gelecek verilen saklanması için sıfırlarla doldurulmuş bir tensör oluşturur
        for j, bbox_pair in enumerate(utils.get_bounding_boxes(label)):     # Her bir bbox ın j=indexi, bbox_pair=sınıf adı ve koordinatlarını içeren (class_name, (x_min, x_max, y_min, y_max)) değerini döndürür
            name, coords = bbox_pair                                        
            assert name in self.classes, f"Unrecognized class '{name}'"     # Sınıf adı self.classes sözlüğünde yoksa "Bilinmeyen sınıf" hatası verir
            class_index = self.classes[name]                                # Sınıfların index ini alır   
            x_min, x_max, y_min, y_max = coords                             # Bbox koordinatlarını alır

            # Yukarıda hesaplanan rastgele kaydırma ve ölçeklendirme değerlerini kullanarak bbox koordinatlarını günceller
            if self.augment:
                half_width = config.IMAGE_SIZE[0] / 2
                half_height = config.IMAGE_SIZE[1] / 2
                x_min = utils.scale_bbox_coord(x_min, half_width, scale) + x_shift
                x_max = utils.scale_bbox_coord(x_max, half_width, scale) + x_shift
                y_min = utils.scale_bbox_coord(y_min, half_height, scale) + y_shift
                y_max = utils.scale_bbox_coord(y_max, half_height, scale) + y_shift

            # Bbox ların merkez noktalarını hesaplar ve hangi grid hücresine ait olduğunu belirler
            mid_x = (x_max + x_min) / 2
            mid_y = (y_max + y_min) / 2
            col = int(mid_x // grid_size_x)     # Merkez x koordinatını grid hücresinin genişliğine bölerek hangi sütunda olduğunu belirler
            row = int(mid_y // grid_size_y)     # Merkez y koordinatını grid hücresinin yüksekliğine bölerek hangi satırda olduğunu belirler
            
            if 0 <= col < config.S and 0 <= row < config.S:                 # Col ve row değerleri 0 ile S (7) arasında ise, yani bbox görselin içinde ise
                cell = (row, col)                                           # Cell adlı bir tuple oluşturur ve bu tuple bbox ın hangi grid hücresine ait olduğunu temsil eder
                if cell not in class_names or name == class_names[cell]:    # Eğer cell daha önce class_names sözlüğünde yoksa veya mevcut sınıf adı zaten o hücreye ait ise
                    
                    # One-hot encoding ile sınıf etiketini ground truth tensörüne ekler
                    one_hot = torch.zeros(config.C)
                    one_hot[class_index] = 1.0
                    ground_truth[row, col, :config.C] = one_hot
                    class_names[cell] = name

                    # Cell hakkı 2 olan bbox sayısını aşmadıysa, bbox bilgilerini ground truth tensörüne ekler
                    bbox_index = boxes.get(cell, 0)                             # Bbox index ini alır, eğer cell daha önce boxes sözlüğünde yoksa 0 olarak başlatır
                    if bbox_index < config.B:
                        bbox_truth = (
                            (mid_x - col * grid_size_x) / config.IMAGE_SIZE[0],  # Tahmin edilen bbox ın x koordinatını grid hücresine göre ekler
                            (mid_y - row * grid_size_y) / config.IMAGE_SIZE[1],  # Tahmin edilen bbox ın y koordinatını grid hücresine göre ekler
                            (x_max - x_min) / config.IMAGE_SIZE[0],  # Genişlik
                            (y_max - y_min) / config.IMAGE_SIZE[1],  # Yükseklik
                            1.0  # Güven skoru 1.0 olarak ayarlanır çünkü bu bbox görselin içinde 
                        )

                        # Mevcut bbox in başlangıç indeksini hesaplar ve ground truth tensörüne bbox bilgilerini ekler
                        # Daha sonra boxes sözlüğünde cell için bbox sayısını 1 artırır
                        bbox_start = 5 * bbox_index + config.C
                        ground_truth[row, col, bbox_start:] = torch.tensor(bbox_truth).repeat(config.B - bbox_index)
                        boxes[cell] = bbox_index + 1

        return data, ground_truth, original_data    # Verileri, tahmin sözlüğünü ve orijinal görseli döndürür

    def __len__(self):
        return len(self.dataset)    # Veri kümesindeki örnek sayısını döndürür