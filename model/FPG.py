import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class DilationOrErosionOperation(nn.Module):
    def __init__(self, device):
        super(DilationOrErosionOperation, self).__init__()

        self.kernels = {
            "left": torch.tensor([[0, 0, 0],
                                  [0, 1, 1],
                                  [0, 0, 0]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0),
            "right": torch.tensor([[0, 0, 0],
                                   [1, 1, 0],
                                   [0, 0, 0]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0),
            "up": torch.tensor([[0, 0, 0],
                                [0, 1, 0],
                                [0, 1, 0]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0),
            "down": torch.tensor([[0, 1, 0],
                                  [0, 1, 0],
                                  [0, 0, 0]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0),
            "center": torch.tensor([[0, 1, 0],
                                    [1, 1, 1],
                                    [0, 1, 0]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0),
            "top_left": torch.tensor([[0, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0),
            "top_right": torch.tensor([[0, 0, 0],
                                       [0, 1, 0],
                                       [1, 0, 0]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0),
            "bottom_left": torch.tensor([[0, 0, 1],
                                         [0, 1, 0],
                                         [0, 0, 0]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0),
            "bottom_right": torch.tensor([[1, 0, 0],
                                          [0, 1, 0],
                                          [0, 0, 0]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        }

    def forward(self, x, operation="erode"):
        # Randomly select a scaling direction.
        direction = random.choice(list(self.kernels.keys()))

        kernel = self.kernels[direction]
        output = F.conv2d(x, kernel, padding=1)

        # Perform dilation or erosion on the input image.
        if operation == "dilate":
            output = (output > 0).float()
        elif operation == "erode":
            output = (output > 1).float()

        return output


class Operation(nn.Module):
    def __init__(self, probability_of_dilatation=0.9, device='cpu'):
        super(Operation, self).__init__()
        self.operation = DilationOrErosionOperation(device)
        self.probability_of_dilatation = probability_of_dilatation

    def forward(self, x, blur_frequency=2):
        for i in range(blur_frequency):
            if random.random() < self.probability_of_dilatation:  # 90%
                operation_type = "dilate"
            else:  # 10%
                operation_type = "erode"

            x = self.operation(x, operation=operation_type)
        return x


class FPG(nn.Module):
    def __init__(self, probability_of_losing_labels=0.2, probability_of_dilatation=0.9, edema_operation_frequency=2,
                 tumor_operation_frequency=5, device='cuda'):
        super(FPG, self).__init__()
        self.probability_of_losing_labels = probability_of_losing_labels
        self.operation = Operation(probability_of_dilatation=probability_of_dilatation, device=device)
        self.edema_operation_frequency = edema_operation_frequency
        self.tumor_operation_frequency = tumor_operation_frequency

    def forward(self, seg_with_2, seg_with_1_and_4):
        if random.random() < self.probability_of_losing_labels:
            return torch.zeros_like(seg_with_2)

        else:
            seg_with_2_dilated = self.operation(seg_with_2, self.edema_operation_frequency) * 0.5
            seg_with_1_and_4_dilated = self.operation(seg_with_1_and_4, self.tumor_operation_frequency)

            output = seg_with_2_dilated
            output[seg_with_1_and_4_dilated == 1] = 1.0

        return output
