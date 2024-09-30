# models/cv/cv_model.py

import torch
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np
from PIL import Image
import os
from mmdet.apis import init_detector, inference_detector
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime


# Initialize the models
CONFIG_FILE = 'carDDModel/dcn_plus_cfg_small.py'
CHECKPOINT_FILE = 'carDDModel/checkpoint.pth'
DEVICE = 'cuda:0'

# Initialize MMDetection model
model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)

# Initialize FasterRCNN
det_model = fasterrcnn_resnet50_fpn(pretrained=True).cuda().eval()
transform = T.ToTensor()


# CV-Related Functions
def get_best_vehicle_box(det_output):
    max_score = 0
    max_bbox = None
    vehicle_classes = [2, 3, 7, 8]  # Car, truck, bus in MS COCO
    for i in range(len(det_output['boxes'])):
        bbox = det_output['boxes'][i]
        score = det_output['scores'][i]
        label = det_output['labels'][i]
        if label in vehicle_classes and score > max_score:
            max_bbox = bbox
            max_score = score
    return max_bbox


def adjust_bbox(bbox, img_width, img_height, margin=0.1):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    x_margin = int(width * margin)
    y_margin = int(height * margin)

    new_x1 = max(0, x1 - x_margin)
    new_y1 = max(0, y1 - y_margin)
    new_x2 = min(img_width, x2 + x_margin)
    new_y2 = min(img_height, y2 + y_margin)

    return [new_x1, new_y1, new_x2, new_y2]


def process_damage_analysis(result, classification_dict, confidence_threshold=0.5):
    damage_info = {classification_dict[i]: [] for i in classification_dict.keys()}
    damage_masks = {classification_dict[i]: [] for i in classification_dict.keys()}

    for classification_label, bboxes in enumerate(result[0], 1):
        damage_type = classification_dict[classification_label]

        for i, bbox in enumerate(bboxes):
            score = bbox[4]
            if score >= confidence_threshold:
                x1, y1, x2, y2 = bbox[:4]
                area = (x2 - x1) * (y2 - y1)

                damage_info[damage_type].append({
                    'index': i + 1,
                    'area': area,
                    'score': score
                })

                if len(result[1]) > classification_label and len(result[1][classification_label]) > i:
                    damage_masks[damage_type].append(result[1][classification_label][i])

    return damage_info, damage_masks


def calculate_repair_cost_analysis(damage_info, car_value, car_age, damage_factors, image_size):
    repair_cost = 0
    total_image_area = image_size[0] * image_size[1]

    car_age = min(car_age, 10)
    age_factor = max(0.5 / 100, (car_age / 10.0))
    chosen_damages = []

    for damage_type, damages in damage_info.items():
        factor = damage_factors[damage_type]
        for damage in damages:
            area_ratio = damage['area'] / total_image_area
            cost = area_ratio * damage['score'] * car_value * factor * age_factor
            repair_cost += cost

            chosen_damages.append({
                'damage_type': damage_type,
                'damage_index': damage['index'],
                'area': damage['area'],
                'score': damage['score'],
                'cost': cost
            })

    return repair_cost, chosen_damages


def visualize_damage_and_costs(damage_info, repair_cost, chosen_damages):
    damage_types = []
    damage_counts = []

    for damage_type, damages in damage_info.items():
        if damages:
            damage_types.append(damage_type.capitalize())
            damage_counts.append(len(damages))

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    ax[0].bar(damage_types, damage_counts, color='skyblue')
    ax[0].set_title('Count of Each Damage Type')
    ax[0].set_xlabel('Damage Type')
    ax[0].set_ylabel('Count')

    damage_costs = {}
    for damage in chosen_damages:
        damage_type = damage['damage_type'].capitalize()
        cost = damage['cost']
        if damage_type in damage_costs:
            damage_costs[damage_type] += cost
        else:
            damage_costs[damage_type] = cost

    ax[1].pie(damage_costs.values(), labels=damage_costs.keys(), autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    ax[1].set_title(f'Repair Cost Breakdown by Damage Type (Total: ${repair_cost:.2f})')

    return fig
