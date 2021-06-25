import matplotlib.pyplot as plt
import numpy as np
import IPython
import torch

from PIL import Image, ImageDraw
from random import randrange
from torchvision.ops import nms


def draw_cell_boundaries(image, cells=7):
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    image_width, image_height = image.size

    fill_color = (255, 255, 128, 200)

    unit = image_width / cells
    for idx in range(1, cells):
        vertical_line_x = unit * idx
        overlay_draw.line([(vertical_line_x, 0), (vertical_line_x, image_height)], fill=fill_color)

    unit = image_height / cells
    for idx in range(1, cells):
        horizontal_line_y = unit * idx
        overlay_draw.line([(0, horizontal_line_y), (image_width, horizontal_line_y)], fill=fill_color)

    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def draw_center_cell_object(image, annotator, annotation, cells=7):
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size

    fill_color = (255, 0, 0, 255)

    for item in annotation:
        (class_id, cell_idx_x, cell_idx_y, cell_pos_x, cell_pos_y, width, height) = item
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        horizontal_unit = image_width / cells
        vertical_unit = image_height / cells

        class_name = annotator.labels[class_id]

        # draw center cell as red color
        cxmin = horizontal_unit * cell_idx_x
        cxmax = horizontal_unit * (cell_idx_x + 1)
        cymin = vertical_unit * cell_idx_y
        cymax = vertical_unit * (cell_idx_y + 1)

        draw.line([(cxmin, cymin), (cxmax, cymin)], fill=fill_color)
        draw.line([(cxmax, cymin), (cxmax, cymax)], fill=fill_color)
        draw.line([(cxmax, cymax), (cxmin, cymax)], fill=fill_color)
        draw.line([(cxmin, cymax), (cxmin, cymin)], fill=fill_color)

        cell_obj_center_x = int(cxmin + (cell_pos_x * image_width))
        cell_obj_center_y = int(cymin + (cell_pos_y * image_height))

        oxmin, oxmax = int(cell_obj_center_x + (width * image_width / 2)), int(
            cell_obj_center_x - (width * image_width / 2))
        oymin, oymax = int(cell_obj_center_y + (height * image_height / 2)), int(
            cell_obj_center_y - (height * image_height / 2))

        draw.ellipse([(cell_obj_center_x - 3, cell_obj_center_y - 3), (cell_obj_center_x + 3, cell_obj_center_y + 3)],
                     fill=(255, 0, 0), width=6)
        draw.text((cell_obj_center_x + 8, cell_obj_center_y - 6), "CLSID: %s" % class_name, fill=fill_color)

        random_color_r, random_color_g, random_color_b = randrange(255), randrange(255), randrange(255)
        overlay_color = (random_color_r, random_color_g, random_color_b, 90)
        overlay_draw.rectangle([oxmin, oymin, oxmax, oymax], fill=overlay_color)  # draw object in random color

        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    return image


def draw_center_cell_object_label(image, annotator, label):
    image_width, image_height = image.size

    fill_color = (255, 0, 0, 255)

    # label -> [25, 7, 7]
    cells = label.shape[1]  # 1, 2 indicates cell count

    for cell_idx_y in range(cells):
        for cell_idx_x in range(cells):
            # ignore predictor with no bbox confidence
            if label[4, cell_idx_y, cell_idx_x] == 0:
                continue

            # (class_id, cell_idx_x, cell_idx_y, cell_pos_x, cell_pos_y, width, height) = item
            current_predictor = label[:, cell_idx_y, cell_idx_x]
            (cell_pos_x, cell_pos_y, width, height) = current_predictor[:4]
            class_id = np.argmax(current_predictor[5:])

            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)

            horizontal_unit = image_width / cells
            vertical_unit = image_height / cells

            class_name = annotator.labels[class_id]

            # draw center cell as red color
            cxmin = horizontal_unit * cell_idx_x
            cxmax = horizontal_unit * (cell_idx_x + 1)
            cymin = vertical_unit * cell_idx_y
            cymax = vertical_unit * (cell_idx_y + 1)

            draw = ImageDraw.Draw(image)
            draw.line([(cxmin, cymin), (cxmax, cymin)], fill=fill_color)
            draw.line([(cxmax, cymin), (cxmax, cymax)], fill=fill_color)
            draw.line([(cxmax, cymax), (cxmin, cymax)], fill=fill_color)
            draw.line([(cxmin, cymax), (cxmin, cymin)], fill=fill_color)

            cell_obj_center_x = (cell_idx_x + cell_pos_x) / cells * image_width
            cell_obj_center_y = (cell_idx_y + cell_pos_y) / cells * image_height

            oxmin, oxmax = int(cell_obj_center_x + (width * image_width / 2)), int(
                cell_obj_center_x - (width * image_width / 2))
            oymin, oymax = int(cell_obj_center_y + (height * image_height / 2)), int(
                cell_obj_center_y - (height * image_height / 2))

            draw.ellipse(
                [(cell_obj_center_x - 3, cell_obj_center_y - 3), (cell_obj_center_x + 3, cell_obj_center_y + 3)],
                fill=(255, 0, 0), width=6)
            draw.text((cell_obj_center_x + 8, cell_obj_center_y - 6), "CLSID: %s" % class_name, fill=fill_color)

            random_color_r, random_color_g, random_color_b = 255, 255, 255
            overlay_color = (random_color_r, random_color_g, random_color_b, 90)
            overlay_draw.rectangle([oxmin, oymin, oxmax, oymax], fill=overlay_color, outline=(0, 0, 255), width=2)  # draw object in random color

            image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    return image


# TODO: redesign based on 2 predictor output
# ignore predictor with lower bbox confidence

# model output version of draw_center_cell_object
def draw_center_cell_object_output(image, annotator, output, confidence_threshold=0.3):
    image_width, image_height = image.size

    # output -> [30, 7, 7]
    cells = output.shape[1]  # 1, 2 indicates cell count
    bboxes = (output.shape[0] - 20) // 5
    assert ((output.shape[0] - 20) % 5 == 0)

    # Organize bboxes for NMS algorithm
    bbox_coordinates = []
    bbox_scores = []
    bbox_classes = []
    for cell_idx_y in range(cells):
        for cell_idx_x in range(cells):
            for bbox_idx in range(bboxes):
                current_predictor = output[5 * (bbox_idx):5 * (bbox_idx + 1), cell_idx_y, cell_idx_x]
                (cell_pos_x, cell_pos_y, width, height, confidence) = torch.sigmoid(
                    torch.from_numpy(current_predictor)).numpy()

                class_prob = torch.sigmoid(torch.from_numpy(output[5 * bboxes:, cell_idx_y, cell_idx_x])).numpy()
                class_id = np.argmax(class_prob)

                if confidence < confidence_threshold:
                    continue

                obj_center_x = (cell_idx_x + cell_pos_x) / cells
                obj_center_y = (cell_idx_y + cell_pos_y) / cells
                oxmin, oxmax = obj_center_x - (width / 2), obj_center_x + (width / 2)
                oymin, oymax = obj_center_y - (height / 2), obj_center_y + (height / 2)

                bbox_coordinates.append([oxmin, oymin, oxmax, oymax])
                bbox_scores.append(class_prob[class_id] * confidence)
                bbox_classes.append(class_id)

    if len(bbox_coordinates) > 0:
        bbox_coordinates = torch.from_numpy(np.array(bbox_coordinates)).float()
        bbox_scores = torch.from_numpy(np.array(bbox_scores)).float()
        bbox_classes = torch.from_numpy(np.array(bbox_classes)).float()

        coordinates_indicies = nms(boxes=bbox_coordinates, scores=bbox_scores, iou_threshold=0.8)

        bbox_filtered_coordinates = bbox_coordinates.index_select(0, coordinates_indicies)
        bbox_filtered_scores = bbox_scores.index_select(0, coordinates_indicies)
        bbox_filtered_classes = bbox_classes.index_select(0, coordinates_indicies).int()

        for idx in range(bbox_filtered_coordinates.shape[0]):
            (xmin, ymin, xmax, ymax) = bbox_filtered_coordinates[idx].numpy()

            oxmin, oxmax = max(xmin * image_width, 0), min(xmax * image_width, image_width - 1)
            oymin, oymax = max(ymin * image_height, 0), min(ymax * image_height, image_height - 1)

            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)

            random_color_r, random_color_g, random_color_b = 0, 255, 255
            overlay_color = (random_color_r, random_color_g, random_color_b, 90)
            overlay_draw.rectangle([oxmin, oymin, oxmax, oymax], fill=overlay_color, outline=(0, 0, 255), width=2)  # draw object in random color

            image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

        for idx in range(bbox_filtered_coordinates.shape[0]):
            (xmin, ymin, xmax, ymax) = bbox_filtered_coordinates[idx].numpy()
            confidence_value = bbox_filtered_scores[idx].numpy() * 100
            class_id = bbox_filtered_classes[idx].numpy()

            oxmin, oxmax = max(xmin * image_width, 0), min(xmax * image_width, image_width - 1)
            oymin, oymax = max(ymin * image_height, 0), min(ymax * image_height, image_height - 1)

            cell_obj_center_x, cell_obj_center_y = int(oxmin + (oxmax - oxmin) / 2), int(oymin + (oymax - oymin) / 2)
            class_name = annotator.labels[class_id]

            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)

            overlay_draw.ellipse(
                [(cell_obj_center_x - 3, cell_obj_center_y - 3), (cell_obj_center_x + 3, cell_obj_center_y + 3)],
                fill=(255, 0, 0, 200), width=6)

            overlay_draw.text((cell_obj_center_x + 8, cell_obj_center_y - 6), "%s\n%.1f%%" % (class_name, confidence_value), fill=(255, 0, 0, 200))
            image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    return image