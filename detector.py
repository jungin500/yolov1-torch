import torch
from torch import nn


class YoloDetector(nn.Module):

    def __init__(
        self,
        cell_size=7,
        box_candidates=2,
        num_classes=20,
        lambda_coord=5,
        lambda_noobj=.5,
    ):
        super(YoloDetector, self).__init__()
        self.cell_size = cell_size
        self.box_candidates = box_candidates
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        grid_indicies = self._generate_grid(cell_size)
        class_one_hot = torch.nn.functional.one_hot(
            torch.arange(0, self.num_classes))

        self.register_buffer('grid_indicies', grid_indicies, persistent=False)
        self.register_buffer('class_one_hot', class_one_hot, persistent=False)

    ''' Generate grids that corresponds to index of each cells
    grid_indicies.shape -> (1, 2, 7, 7)
    e.g: grid_indicies[B, :, 3, 6] -> tensor([6, 3])
    '''

    def _generate_grid(self, cell_size):
        indicies = torch.arange(0, cell_size).repeat((cell_size, 1))
        grid = torch.cat([indicies.unsqueeze(2), indicies.T.unsqueeze(2)], 2)
        return grid.transpose(0, 2).transpose(1, 2).unsqueeze(0)

    def infer(self, x):
        boxes = torch.Tensor().to(x)
        scores = torch.Tensor().to(x)
        idxs = torch.Tensor().to(x)

        for idx in range(self.box_candidates):
            det_box = x[:, (idx) * 5:(idx + 1) * 5, :, :]

            det_cls_flatten = torch.flatten(x[:, -self.num_classes:, :, :] *
                                            det_box[:, 4, None, :, :],
                                            start_dim=2)

            det_clsid = det_cls_flatten.argmax(1)  # [B, 49]
            det_conf = det_cls_flatten.max(dim=1).values  # [B, 49]

            det_box_xycenter_rel = (det_box[:, :2, :, :] +
                                    self.grid_indicies) / self.cell_size
            det_box_xymin = det_box_xycenter_rel - (det_box[:, 2:4, :, :] / 2)
            det_box_xymax = det_box_xycenter_rel + (det_box[:, 2:4, :, :] / 2)

            det_box_xymin = torch.flatten(det_box_xymin,
                                          start_dim=2).transpose(1, 2)
            det_box_xymax = torch.flatten(det_box_xymax,
                                          start_dim=2).transpose(1, 2)

            det_box_xyxy = torch.cat([det_box_xymin, det_box_xymax],
                                     dim=2)  # [B, 49, 4]

            boxes = torch.cat([boxes, det_box_xyxy], dim=1)
            scores = torch.cat([scores, det_conf], dim=1)
            idxs = torch.cat([idxs, det_clsid], dim=1)

        return boxes, scores, idxs

    def get_loss(self, x, batched_labels):
        # TODO: move to dataloader
        gt_box = torch.zeros((
            x.shape[0],
            5 + self.num_classes,
            x.shape[2],
            x.shape[3],
        ))

        for batch_idx, labels in enumerate(batched_labels):
            for xcenter, ycenter, width, height, class_id in labels:
                grid_x = int(xcenter * self.cell_size)
                grid_y = int(ycenter * self.cell_size)
                xcenter_rel = xcenter * self.cell_size - grid_x
                ycenter_rel = ycenter * self.cell_size - grid_y
                class_probs = self.class_one_hot[int(class_id)]
                gt_box[batch_idx, :, grid_x, grid_y] = torch.cat([
                    torch.Tensor([
                        xcenter_rel,
                        ycenter_rel,
                        width,
                        height,
                        1,  # 'Has Object' flag
                    ]).to(x),
                    class_probs
                ])
        gt_box = gt_box.to(x)
        # End TODO

        gt_box_mask = gt_box[:, 4, None, :, :]

        # Calculate IoU between GT and Det
        ious = []
        for idx in range(self.box_candidates):
            det_box = x[:, (idx) * 5:(idx + 1) * 5, :, :]

            # from torchvision.ops.boxes import box_iou
            # Get IoU between det_box and gt_boxes
            # implementation from: _box_inter_union
            det_box_area = det_box[:, 2, :, :] * det_box[:, 3, :, :]
            gt_box_area = gt_box[:, 2, :, :] * gt_box[:, 3, :, :]

            det_box_xycenter_rel = (det_box[:, :2, :, :] +
                                    self.grid_indicies) / self.cell_size
            det_box_xymin = det_box_xycenter_rel - (det_box[:, 2:4, :, :] / 2)
            det_box_xymax = det_box_xycenter_rel + (det_box[:, 2:4, :, :] / 2)

            gt_box_xycenter_rel = (gt_box[:, :2, :, :] +
                                   self.grid_indicies) / self.cell_size
            gt_box_xymin = gt_box_xycenter_rel - (gt_box[:, 2:4, :, :] / 2)
            gt_box_xymax = gt_box_xycenter_rel + (gt_box[:, 2:4, :, :] / 2)

            lt = torch.max(det_box_xymax.unsqueeze(1), gt_box_xymax)
            rb = torch.min(det_box_xymin.unsqueeze(1), gt_box_xymin)
            wh = (rb - lt).clamp(min=0)
            inter = wh[:, :, 0, :, :] * wh[:, :, 1, :, :]
            union = det_box_area[:, None, :, :] + gt_box_area - inter
            iou = inter / union  # B, 1, 7, 7
            ious.append(iou)

        ious = torch.cat(ious, dim=1)  # B, 2, 7, 7

        loss = .0
        ious_argmax = ious.argmax(dim=1, keepdim=True)

        class_det = x[:, -self.num_classes:, :, :]
        class_gt = gt_box[:, -self.num_classes:, :, :]

        for box_index in range(ious.shape[1]):
            # is_candidate mask for box index
            iou_mask = ious_argmax == box_index
            obj_mask = (iou_mask * gt_box_mask).bool()

            det_box = x[:, (idx) * 5:(idx + 1) * 5, :, :]
            # gt_box can be used without selecting candidates (mask required though)

            xy_loss = self.lambda_coord * torch.sum(
                torch.pow(det_box[:, :2, :, :] - gt_box[:, :2, :, :], 2) *
                obj_mask)

            w_loss = torch.pow(
                torch.sqrt(det_box[:, 2, :, :]) -
                torch.sqrt(gt_box[:, 2, :, :]), 2)
            h_loss = torch.pow(
                torch.sqrt(det_box[:, 3, :, :]) -
                torch.sqrt(gt_box[:, 3, :, :]), 2)
            wh_loss = self.lambda_coord * torch.sum(
                (w_loss[:, None, :, :] + h_loss[:, None, :, :]) * obj_mask)

            conf_det = torch.sum(class_det * class_gt,
                                 1) * ious[:, box_index, :, :]
            conf_gt = ious[:, box_index, :, :]

            conf_loss = torch.sum((conf_det - conf_gt) * obj_mask)
            conf_noobj_loss = self.lambda_noobj * torch.sum(
                (conf_det - conf_gt) * (~obj_mask))

            loss += xy_loss + wh_loss + conf_loss + conf_noobj_loss

        class_loss = torch.sum(
            torch.pow(class_det - class_gt, 2) * gt_box_mask)
        loss += class_loss

        return loss

    def forward(self, x, batched_labels=None):
        assert x.shape[1] == self.num_classes + self.box_candidates * 5 \
            and x.shape[2] == self.cell_size \
            and x.shape[3] == self.cell_size

        if self.training:
            pred = self.infer(x)
            loss = self.get_loss(x, batched_labels)
            return pred, loss
        else:
            return self.infer(x)


if __name__ == '__main__':
    import numpy as np

    criterion = YoloDetector()
    criterion.to('cuda')
    criterion.train()

    pseudo_det_result = torch.zeros(1, 30, 7, 7)
    pseudo_det_result[:, :5, :, :] = torch.from_numpy(
        np.array([0., 1., 2., 3.,
                  4.])).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    pseudo_det_result[:, 5:10, :, :] = torch.from_numpy(
        np.array([0., 1., 2., 3.,
                  4.])).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    pseudo_det_result[:, 10:, :, :] = torch.from_numpy(
        np.linspace(0, 19, 20) + 100).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    pseudo_det_result = pseudo_det_result.to('cuda')

    labels = []
    labels.append([0.4, 0.6, 0.3, 0.4, 4])
    labels.append([0.3, 0.4, 0.1, 0.1, 2])

    batched_labels = torch.Tensor([labels]).to('cuda')

    # batched_labels: xcenter, ycenter, width, height, class_id
    pred, loss = criterion(pseudo_det_result, batched_labels)
    print("Loss: {:.04f}".format(loss))

    criterion.eval()
    pred = criterion(pseudo_det_result)
    print("Pred:", pred)  # boxes, scores, idxs