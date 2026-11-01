import torch
import json
import os
import config
import matplotlib.patches as patches
import torchvision.transforms as T
from PIL import ImageDraw, ImageFont
from matplotlib import pyplot as plt


def get_iou(p, a):
    p_tl, p_br = bbox_to_coords(p)          # (batch, S, S, B, 2)
    a_tl, a_br = bbox_to_coords(a)

    # Burada iki tensorun elemanlarını karşılaştırıyoruz. Her koordinat için büyük olanı seçiyoruz.
    coords_join_size = (-1, -1, -1, config.B, config.B, 2)
    tl = torch.max(
        p_tl.unsqueeze(4).expand(coords_join_size),         # (batch, S, S, B, 1, 2) -> (batch, S, S, B, B, 2)
        a_tl.unsqueeze(3).expand(coords_join_size)          # (batch, S, S, 1, B, 2) -> (batch, S, S, B, B, 2)
    )
    br = torch.min(
        p_br.unsqueeze(4).expand(coords_join_size),
        a_br.unsqueeze(3).expand(coords_join_size)
    )

    # Burada tahmin ile GT bbox ların kesişim alanını hesaplıyoruz. Eğer bbox lar kesişmiyorsa, intersection_sides negatif olur ve torch.clamp ile 0 a eşitleriz.
    intersection_sides = torch.clamp(br - tl, min=0.0)
    intersection = intersection_sides[..., 0] \
                   * intersection_sides[..., 1]       # (batch, S, S, B, B)


    # Burada GT ve tahminlerin alanlarını hesaplıyoruz. Daha sonra kesişim alanını kullanarak IOU değerini hesaplıyoruz.
    p_area = bbox_attr(p, 2) * bbox_attr(p, 3)                  # (batch, S, S, B)
    p_area = p_area.unsqueeze(4).expand_as(intersection)        # (batch, S, S, B, 1) -> (batch, S, S, B, B)

    a_area = bbox_attr(a, 2) * bbox_attr(a, 3)                  # (batch, S, S, B)
    a_area = a_area.unsqueeze(3).expand_as(intersection)        # (batch, S, S, 1, B) -> (batch, S, S, B, B)

    union = p_area + a_area - intersection

    # Matematikteki 0'a bölme hatasını önler
    zero_unions = (union == 0.0)
    union[zero_unions] = config.EPSILON
    intersection[zero_unions] = 0.0

    return intersection / union


def bbox_to_coords(t):
    # Bu fonksiyon bbox formatındaki verileri (x, y, width, height)  yerine ([x1,y1], [x2,y2]) formatına çevirir. 

    width = bbox_attr(t, 2)
    x = bbox_attr(t, 0)
    x1 = x - width / 2.0
    x2 = x + width / 2.0

    height = bbox_attr(t, 3)
    y = bbox_attr(t, 1)
    y1 = y - height / 2.0
    y2 = y + height / 2.0

    return torch.stack((x1, y1), dim=4), torch.stack((x2, y2), dim=4)


def scheduler_lambda(epoch):
    # Bu fonskiyon ile lr gate i makaledeki gibi ayarlıyoruz. 
    if epoch < config.WARMUP_EPOCHS + 75:
        return 1
    elif epoch < config.WARMUP_EPOCHS + 105:
        return 0.1
    else:
        return 0.01


def load_class_dict():
    # Sınıf sözlüğünü yükler. Eğer sözlük yoksa boş bir sözlük oluşturur ve kaydeder.

    if os.path.exists(config.CLASSES_PATH):
        with open(config.CLASSES_PATH, 'r', encoding='utf-8') as file:
            return json.load(file)

    new_dict = {}
    save_class_dict(new_dict)
    return new_dict


def load_class_array():
    # Sınıf isimlerini bir liste olarak döndürür. Sınıf sözlüğünü yükler ve index sırasına göre sınıf isimlerini bir listeye ekler.
    classes = load_class_dict()
    result = [None for _ in range(len(classes))]

    for i, c in classes.items():
        result[int(i)] = c

    return result

def save_class_dict(obj):
    # Eğer sınıf sözlüğü yoksa, belirtilen yolda bir klasör oluşturur ve sözlüğü kaydeder.
    folder = os.path.dirname(config.CLASSES_PATH)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(config.CLASSES_PATH, 'w') as file:
        json.dump(obj, file, indent=2)


def get_dimensions(label):
    # Width ve height değerlerini label dosyasından alır. Bu değerler bbox koordinat hesaplamalarında kullanılır.
    size = label['annotation']['size']
    return int(size['width']), int(size['height'])


def get_bounding_boxes(label):
    # Koordinatları alınan bbox ların sınıf isimleri ile birlikte bir liste olarak döndürür. Koordinatlar görselin boyutuna göre normalize edilir.

    width, height = get_dimensions(label)
    x_scale = config.IMAGE_SIZE[0] / width
    y_scale = config.IMAGE_SIZE[1] / height
    boxes = []
    objects = label['annotation']['object']
    for obj in objects:
        box = obj['bndbox']
        coords = (
            int(int(box['xmin']) * x_scale),
            int(int(box['xmax']) * x_scale),
            int(int(box['ymin']) * y_scale),
            int(int(box['ymax']) * y_scale)
        )
        name = obj['name']
        boxes.append((name, coords))
    return boxes


def bbox_attr(data, i):
    # Hesaplama için oluşturdıuğummuz vektörün ilk 20 değeri sınıf bilgisi, sonraki 5 değer bbox bilgisi, sonraki 5 değer diğer bbox bilgisi şeklinde sıralanır. 
    # Bu yüzden vektörde ulaşmak isteediğimiz değere ulaşmak için sınıf sayısı kadar offset eklememiz gerekir. Örneğin, 0. bbox'ın x koordinatına ulaşmak için 20 + 0 = 20. indeksi kullanırız. 
    # 1. bbox'ın x koordinatına ulaşmak için 20 + 5 + 0 = 25. indeksi kullanırız.

    attr_start = config.C + i               # İlgili bbox değerine ulaşmak için sınıf bilgisi engelini ortadan kaldırmaya yardım eder
    return data[..., attr_start::5]         # Örüntü 5 er li ilerlediğinde için bir sonraki değer için 5 er li adım atarız. 
                                            # Örneğin, 0. bbox'ın x koordinatına ulaşmak için 20 + 0 = 20. indeksi kullanırız. 1. bbox'ın x koordinatına ulaşmak için 20 + 5 + 0 = 25. indeksi kullanırız.


def scale_bbox_coord(coord, center, scale):
    return ((coord - center) * scale) + center


def get_overlap(a, b):
# İki kutu arasındaki örtüşme oranını (sol-üst köşe, genişlik, yükseklik, güven skoru, sınıf) biçiminde döndürür.

    a_tl, a_width, a_height, _, _ = a
    b_tl, b_width, b_height, _, _ = b

    i_tl = (
        max(a_tl[0], b_tl[0]),
        max(a_tl[1], b_tl[1])
    )
    i_br = (
        min(a_tl[0] + a_width, b_tl[0] + b_width),
        min(a_tl[1] + a_height, b_tl[1] + b_height),
    )

    intersection = max(0, i_br[0] - i_tl[0]) \
                   * max(0, i_br[1] - i_tl[1])

    a_area = a_width * a_height
    b_area = b_width * b_height

    a_intersection = b_intersection = intersection
    if a_area == 0:
        a_intersection = 0
        a_area = config.EPSILON
    if b_area == 0:
        b_intersection = 0
        b_area = config.EPSILON

    return torch.max(
        a_intersection / a_area,
        b_intersection / b_area
    ).item()


def plot_boxes(data, labels, classes, color='orange', min_confidence=0.2, max_overlap=0.5, file=None):
    # Verilen görüntü üzerinde bbox çizer

    grid_size_x = data.size(dim=2) / config.S
    grid_size_y = data.size(dim=1) / config.S
    m = labels.size(dim=0)
    n = labels.size(dim=1)

    bboxes = []
    for i in range(m):
        for j in range(n):
            for k in range((labels.size(dim=2) - config.C) // 5):
                bbox_start = 5 * k + config.C
                bbox_end = 5 * (k + 1) + config.C
                bbox = labels[i, j, bbox_start:bbox_end]
                class_index = torch.argmax(labels[i, j, :config.C]).item()
                confidence = labels[i, j, class_index].item() * bbox[4].item()          # pr(c) * IOU
                if confidence > min_confidence:
                    width = bbox[2] * config.IMAGE_SIZE[0]
                    height = bbox[3] * config.IMAGE_SIZE[1]
                    tl = (
                        bbox[0] * config.IMAGE_SIZE[0] + j * grid_size_x - width / 2,
                        bbox[1] * config.IMAGE_SIZE[1] + i * grid_size_y - height / 2
                    )
                    bboxes.append([tl, width, height, confidence, class_index])

    # Güven skoru en yüksek olan bbox ları önce çizmek için bboxes listesini güven skoruna göre azalan sırada sıralarız.
    bboxes = sorted(bboxes, key=lambda x: x[3], reverse=True)

    # Her bir bbox için diğer bbox lar ile örtüşme oranını hesaplar.
    num_boxes = len(bboxes)
    iou = [[0 for _ in range(num_boxes)] for _ in range(num_boxes)]
    for i in range(num_boxes):
        for j in range(num_boxes):
            iou[i][j] = get_overlap(bboxes[i], bboxes[j])

    # Maksimum olmayan değerlerin bastırılması ve görüntünün oluşturulması
    image = T.ToPILImage()(data)
    draw = ImageDraw.Draw(image)
    discarded = set()
    for i in range(num_boxes):
        if i not in discarded:
            tl, width, height, confidence, class_index = bboxes[i]

            # Çakışan diğer bbox lar elenir
            for j in range(num_boxes):
                other_class = bboxes[j][4]
                if j != i and other_class == class_index and iou[i][j] > max_overlap:
                    discarded.add(j)

            # Bbox çizilir ve sınıf adı ile güven skoru yazılır
            draw.rectangle((tl, (tl[0] + width, tl[1] + height)), outline='orange')
            text_pos = (max(0, tl[0]), max(0, tl[1] - 11))
            text = f'{classes[class_index]} {round(confidence * 100, 1)}%'
            text_bbox = draw.textbbox(text_pos, text)
            draw.rectangle(text_bbox, fill='orange')
            draw.text(text_pos, text)
    if file is None:
        image.show()
    else:
        output_dir = os.path.dirname(file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not file.endswith('.png'):
            file += '.png'
        image.save(file)
