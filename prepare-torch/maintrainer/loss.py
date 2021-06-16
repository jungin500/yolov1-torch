from torch import Tensor
from torch.nn import *
import torch


class YoloLoss(Module):
    def __init__(self, lambda_coord, lambda_noobj):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.eps = 1e-15  # for sqrt(0), we will do sqrt(0 + eps)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert ((input.shape[1] - 20) % 5 == 0)
        bboxes = (input.shape[1] - 20) // 5

        # applies outer-model sigmoid function
        input = torch.sigmoid(input)

        # target: [cell_pos_x, cell_pos_y, width, height, 1.0] * 7 * 7
        object_gt_exist_mask = target[:, 4:5, :, :] == 1
        responsible_bbox_index_mask = YoloLoss.get_responsible_bbox_predictor_mask(input, target, bboxes)

        loss = 0
        for bbox_id in range(bboxes):
            current_box = input[:, 5 * bbox_id:5 * (bbox_id + 1), :, :]

            # we should use with object_gt_exist_mask
            # because non-object-existant mask will have zero index
            current_box_responsible_mask = responsible_bbox_index_mask == bbox_id

            # ! TODO: iterate over xy_loss and find out that object_gt_exist_mask works well
            xy_loss = torch.square(
                current_box[:, :2, :, :] - target[:, :2, :, :]) * object_gt_exist_mask * current_box_responsible_mask
            loss += self.lambda_coord * torch.sum(xy_loss)
            # print("xy_loss ", torch.sum(xy_loss))

            wh_loss = torch.square(torch.sqrt(current_box[:, 2:4, :, :] + self.eps) - torch.sqrt(
                target[:, 2:4, :, :] + self.eps)) * object_gt_exist_mask * current_box_responsible_mask
            loss += self.lambda_coord * torch.sum(wh_loss)
            # print("wh_loss: ", torch.sum(wh_loss))
            # print("wh_loss current_box sqrt", torch.sum(torch.sqrt(current_box[:, 2:4, :, :] + self.eps)))
            # print("wh_loss target sqrt", torch.sum(torch.sqrt(target[:, 2:4, :, :] + self.eps)))

            conf_obj_loss = torch.square(
                current_box[: 4:5, :, :] - target[:, 4:5, :, :]) * object_gt_exist_mask * current_box_responsible_mask
            conf_noobj_loss = torch.square(current_box[: 4:5, :, :] - target[:, 4:5, :, :]) * ~(
                        object_gt_exist_mask * current_box_responsible_mask)
            loss += torch.sum(conf_obj_loss)
            loss += self.lambda_noobj * torch.sum(conf_noobj_loss)
            # print("conf_obj_loss: ", torch.sum(conf_obj_loss))
            # print("conf_noobj_loss: ", torch.sum(conf_noobj_loss))

        class_loss = torch.square(input[:, (5 * bboxes):, :, :] - target[:, 5:, :, :])
        loss += torch.sum(class_loss)
        # print("class_loss: ", torch.sum(class_loss))

        return loss

    @staticmethod
    def get_responsible_bbox_predictor_mask(input: Tensor, target: Tensor, bboxes: int) -> Tensor:
        ious = []
        for bbox_id in range(bboxes):
            current_box_xywh = input[:, 5 * bbox_id:5 * (bbox_id + 1) - 1, :, :]
            label_xywh = target[:, :4, :, :]

            # iou -> (B * 7 * 7)
            iou = YoloLoss.get_iou_xywh(current_box_xywh, label_xywh)
            ious.append(iou)

        # stacked_iou -> (B * 2 * 7 * 7)
        stacked_iou = torch.stack(ious, dim=1)
        # print(stacked_iou.shape, torch.argmax(stacked_iou, dim=1, keepdim=True).shape)

        return torch.argmax(stacked_iou, dim=1, keepdim=True)

    @staticmethod
    def get_iou_xywh(input_xywh: Tensor, label_xywh: Tensor) -> Tensor:
        # index_map -> [1, 2, 7, 7]
        index_map_x = torch.arange(0, 7).repeat(7)
        index_map_y = torch.repeat_interleave(torch.arange(0, 7), 7)
        index_map = torch.unsqueeze(torch.stack([index_map_y, index_map_x], dim=0).view(2, 7, 7), 0)

        if input_xywh.device.type == 'cuda':
            index_map = index_map.cuda(non_blocking=True)

        input_xy_global = (input_xywh[:, :2, :, :] + index_map) / 7
        input_width_half, input_height_half = (input_xywh[:, 2, :, :] / 2), (input_xywh[:, 3, :, :] / 2)
        input_xmin = input_xy_global[:, 0, :, :] - input_width_half  # x_center - width / 2
        input_xmax = input_xy_global[:, 0, :, :] + input_width_half
        input_ymin = input_xy_global[:, 1, :, :] - input_height_half
        input_ymax = input_xy_global[:, 1, :, :] + input_height_half

        label_xy_global = (label_xywh[:, :2, :, :] + index_map) / 7
        label_width_half, label_height_half = (label_xywh[:, 2, :, :] / 2), (label_xywh[:, 3, :, :] / 2)
        label_xmin = label_xy_global[:, 0, :, :] - label_width_half  # x_center - width / 2
        label_xmax = label_xy_global[:, 0, :, :] + label_width_half
        label_ymin = label_xy_global[:, 1, :, :] - label_height_half
        label_ymax = label_xy_global[:, 1, :, :] + label_height_half

        input_volume = input_xywh[:, 2, :, :] * input_xywh[:, 3, :, :]
        label_volume = label_xywh[:, 2, :, :] * label_xywh[:, 3, :, :]
        intersect_width = torch.minimum(input_xmax, label_xmax) - torch.maximum(input_xmin, label_xmin)
        intersect_height = torch.minimum(input_ymax, label_ymax) - torch.maximum(input_ymin, label_ymin)
        intersect_volume = intersect_width * intersect_height
        union_volume = input_volume + label_volume - intersect_volume

        return intersect_volume / union_volume