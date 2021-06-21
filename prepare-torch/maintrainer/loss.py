from torch import Tensor
from torch.nn import *
from torch.nn.functional import *
import torch


class YoloLoss(Module):
    def __init__(self, lambda_coord, lambda_noobj, debug=False):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.eps = 1e-15  # for sqrt(0), we will do sqrt(0 + eps)
        self.debug = debug

        print("NewYoloLossV9 (CUDA not supported for now)")

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert ((input.shape[1] - 20) % 5 == 0)
        bbox_count = (input.shape[1] - 20) // 5

        # target: [cell_pos_x, cell_pos_y, width, height, 1.0] * 7 * 7
        object_gt_exist_mask = target[:, 4:5, :, :] == 1
        # input = torch.cat([torch.sigmoid(input[:, :bbox_count * 5, :, :]), input[:, :20, :, :]], dim=1)
        input = torch.cat([target[:, :5, :, :], target[:, :5, :, :], target[:, -20:, :, :]], dim=1)
        
        ## !TODO Important: we only support 2 bbox predicators!
        resp_indicies, nonresp_indicies = YoloLoss.get_responsible_bbox_predictor_indicies(input, target, bbox_count)  # [B * 1 * 7 * 7]
        resp_indicies = torch.unsqueeze(resp_indicies, 1).repeat(1, 1, 5, 1, 1)  # [B * 1 * 5 * 7 * 7]
        nonresp_indicies = torch.unsqueeze(nonresp_indicies, 1).repeat(1, 1, 5, 1, 1)  # [B * 1 * 5 * 7 * 7]

        bboxes = []
        for bbox_id in range(bbox_count):
            current_box = input[:, 5 * bbox_id:5 * (bbox_id + 1), :, :]
            bboxes.append(torch.unsqueeze(current_box, 1))
        merged_bbox = torch.cat(bboxes, dim=1)  # [B * 2 * 5 * 7 * 7]

        loss = 0

        print("\t* First predictor: ", torch.sum(resp_indicies == 0), ", Second predictor: ", torch.sum(resp_indicies == 1))

        responsible_predictor = torch.squeeze(torch.gather(merged_bbox, 1, resp_indicies), 1)
        nonresp_predictor = torch.squeeze(torch.gather(merged_bbox, 1, nonresp_indicies), 1)

        # # ! TODO: iterate over xy_loss and find out that object_gt_exist_mask works well
        xy_loss = torch.square(responsible_predictor[:, :2, :, :] - target[:, :2, :, :]) * object_gt_exist_mask
        loss += self.lambda_coord * torch.sum(xy_loss)
        if self.debug: print("\nxy_loss ", torch.sum(xy_loss))
#         for idx in range(input.shape[0]):
#             print("idx[%d] xy_loss: " % idx, torch.sum(xy_loss[idx, :, :, :]).clone().detach().cpu().numpy().item(), end=' ')
#         print()

        wh_loss = torch.square(torch.sqrt(responsible_predictor[:, 2:4, :, :]) - torch.sqrt(target[:, 2:4, :, :])) * object_gt_exist_mask
        loss += self.lambda_coord * torch.sum(wh_loss)
        if self.debug: print("wh_loss: ", torch.sum(wh_loss))

        print("** Shape currently you want to know: ", torch.sum(torch.square(target[:, 4:5, :, :] - target[:, 4:5, :, :]) * object_gt_exist_mask))

        confidence_difference = torch.square(target[:, 4:5, :, :] - target[:, 4:5, :, :])
        conf_obj_loss = confidence_difference * object_gt_exist_mask
        conf_noobj_loss = confidence_difference * ~object_gt_exist_mask
        loss += torch.sum(conf_obj_loss)
        loss += self.lambda_noobj * torch.sum(conf_noobj_loss)

        if self.debug: print("conf_obj_loss: ", torch.sum(conf_obj_loss))
        if self.debug: print("conf_noobj_loss: ", torch.sum(conf_noobj_loss))

        # view every boxes classes as a batch
        # collapsed [B, 20, 7, 7] -> [B*items, 20]
        masked_input_classes = torch.masked_select(input=input[:, -20:, :, :], mask=object_gt_exist_mask).view(-1, 20)
        masked_target_classes = torch.masked_select(input=target[:, -20:, :, :], mask=object_gt_exist_mask).view(-1, 20)
        assert(masked_input_classes.shape[0] == masked_target_classes.shape[0])  # MUST be same!

        # CE-version Loss for classes (improved)
        # print("\t*", masked_target_classes.shape)
        # ce_loss = cross_entropy(masked_input_classes, torch.argmax(masked_target_classes, 1))
        # loss += ce_loss

        # MSE-version Loss for classes (paper)
        class_loss = torch.square(input[:, -20:, :, :] - target[:, -20:, :, :]) * object_gt_exist_mask
        loss += torch.sum(class_loss)
        if self.debug: print("class_loss: ", torch.sum(class_loss))

        # if self.debug: print("** total loss: ", loss, " **")
        
        return loss


    
    @staticmethod
    def get_responsible_bbox_predictor_indicies(input: Tensor, target: Tensor, bboxes: int) -> Tensor:
        ious = []
        for bbox_id in range(bboxes):
            current_box_xywh = input[:, 5 * bbox_id:5 * (bbox_id + 1) - 1, :, :]
            label_xywh = target[:, :4, :, :]

            iou = YoloLoss.get_iou_xywh(current_box_xywh, label_xywh)
            iou = torch.unsqueeze(iou, 1)  # iou -> (B * 1 * 7 * 7)
            ious.append(iou)

        stacked_iou = torch.stack(ious, dim=1)  # stacked_iou -> (B * 2 * 7 * 7)
        return torch.argmax(stacked_iou, dim=1), torch.argmin(stacked_iou, dim=1)

    @staticmethod
    def get_iou_xywh(input_xywh: Tensor, label_xywh: Tensor) -> Tensor:
        # index_map -> [1, 2, 7, 7]
        index_map_x = torch.arange(0, 7, device=input_xywh.device).repeat(7)
        index_map_y = torch.repeat_interleave(torch.arange(0, 7, device=input_xywh.device), 7)
        index_map = torch.unsqueeze(torch.stack([index_map_y, index_map_x], dim=0).view(2, 7, 7), 0)
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