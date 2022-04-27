# -*- coding: utf-8 -*-

import torch
import json
import numpy as np
from utils import draw_shadow
from utils import shadow_edge_blur
from utils import image_transformation
from utils import random_param_generator
from utils import polygon_correction
from torchvision.transforms import Compose

with open('params.json', 'r') as config:
    params = json.load(config)
    device = params['device']


class Particle:
    r"""
    Particles in PSO.

    Args:
        dim: Number of parameters to be optimized.
        coord_min: The minimum coordinate values.
        coord_max: The maximum coordinate values.
        max_speed: The maximum speed of a particle.
    """
    def __init__(
        self,
        dim: int = 6,
        coord_min: float = -16.,
        coord_max: float = 48.,
        max_speed: float = 1.5
    ) -> None:
        self.pos = polygon_correction(np.random.uniform(coord_min, coord_max, dim))
        self.speed = np.random.uniform(-max_speed, max_speed, dim)
        self.best_pos = np.zeros(dim)
        self.fitness_value = float('inf')


class PSO:
    r"""
    Particle Swarm optimization.

    Args:
        dim: Number of parameters to be optimized.
        size: Number of particles.
        iter_num: The maximum number of iterations.
        coord_min: The minimum coordinate values.
        coord_max: The maximum coordinate values.
        max_speed: The maximum speed of a particle.
        coefficient: The shadow coefficient :math:`k`.
        label: The ground-truth label of the image.
        image: The image to be attacked with shape :math:`(3, H, W)`.
        coord: The coordinates of the points where mask == 1.
        model: The model to be attacked.
        targeted: Targeted / Non-targeted attack.
        physical: Physical / digital attack.
        pre_process: Pre-processing operations on the image.
    """
    def __init__(
        self,
        dim: int = 6,
        size: int = 10,
        iter_num: int = 100,
        coord_min: float = -16.,
        coord_max: float = 48.,
        max_speed: float = 1.5,
        coefficient: float = .43,
        image: torch.Tensor = None,
        coord: torch.Tensor = None,
        model: torch.nn.Module = None,
        targeted: bool = False,
        physical: bool = False,
        label: int = 0,
        pre_process: Compose = Compose([]),
        **parameters
    ) -> None:
        self.w = 1                  # Inertia weight of PSO.
        self.c1 = self.c2 = 2       # Acceleration coefficient of PSO.
        self.dim = dim
        self.size = size
        self.iter_num = iter_num
        self.coord_min = coord_min
        self.coord_max = coord_max
        self.max_speed = max_speed
        self.coefficient = coefficient
        self.label = label
        self.image = image
        self.coord = coord
        self.model = model
        self.targeted = targeted
        self.physical = physical
        self.pre_process = pre_process
        self.best_fitness_value = float('inf')
        self.best_position = np.zeros(dim)
        self.parameters = parameters
        self.succeed = False
        self.num_query = 0

        # Population_initialization
        self.Particle_list = [
            Particle(dim, coord_min, coord_max, max_speed) for _ in range(size)]

    def fit_fun(self, position, **parameters):
        r"""
        Fitness function in PSO.

        Args:
            position: The coordinates of the vertices of the polygonal shadow area.

        Returns:
            confidence: The lower the value, the more successful the attack.
            success: Whether the attack is successful.
            predict: The model output.
        """

        if self.physical:
            img = image_transformation(
                self.image, position, self.coord,
                *parameters.get('rand_param'), self.pre_process).to(device)
        else:
            img, shadow_area = draw_shadow(position, self.image, self.coord, self.coefficient)
            img = shadow_edge_blur(img, shadow_area, 3)
            img = self.pre_process(img).unsqueeze(0).to(device)

        with torch.no_grad():
            predict = torch.softmax(self.model(img), 1)
            predict = torch.mean(predict, dim=0)

        if self.targeted:
            target = parameters.get("target")
            confidence = float(1 - predict[target])
            success = torch.argmax(predict) == target
        else:
            confidence = float(predict[self.label])
            success = torch.argmax(predict) != self.label

        self.num_query += img.shape[0]
        return confidence, success, predict

    def update_speed(self, part):
        speed_value = self.w * part.speed \
                    + self.c1 * np.random.uniform(self.dim) * (part.best_pos - part.pos) \
                    + self.c2 * np.random.uniform(self.dim) * (self.best_position - part.pos)
        part.speed = speed_value.clip(-self.max_speed, self.max_speed)

    def update_pos(self, part, **parameters):
        part.pos = polygon_correction(
            (part.pos + part.speed).clip(self.coord_min, self.coord_max))
        parameters.update(self.parameters)
        value, succeed, _ = self.fit_fun(part.pos, **parameters)
        self.succeed |= succeed
        if value < part.fitness_value:
            part.fitness_value = value
            part.best_pos = part.pos
        if value < self.best_fitness_value or (succeed and not self.physical):
            self.best_fitness_value = value
            self.best_position = part.pos

    def update_digital(self):
        # Run the PSO algorithm for digital attack.
        for _ in range(self.iter_num):
            for part in self.Particle_list:
                if self.succeed:
                    break
                self.update_speed(part)
                self.update_pos(part)

        return self.best_fitness_value, self.best_position, self.succeed, self.num_query

    def update_physical(self):
        # Run the PSO algorithm for physical attack.
        num = self.parameters.get("transform_num")
        h, w, _ = self.image.shape

        for i in range(self.iter_num):
            for part in self.Particle_list:
                self.update_speed(part)
                _num = num * (i < self.iter_num - 100)
                self.update_pos(part, rand_param=random_param_generator(_num, w, h))
            print(f"iteration: {i + 1} {self.best_fitness_value}")

        return self.best_fitness_value, self.best_position, self.succeed, self.num_query
