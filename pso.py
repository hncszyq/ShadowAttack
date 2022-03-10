# -*- coding: utf-8 -*-

import torch
import json
import numpy as np
import random
from utils import draw_shadow
from utils import shadow_edge_blur
from utils import image_transformation
from utils import random_param_generator
from utils import polygon_correction
from torchvision import transforms
from gtsrb import pre_process_image

with open('params.json', 'r') as config:
    params = json.load(config)
    device = params['device']

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(tuple(mean), tuple(std))
])

_shadow_level = None


def fit_fun(position, img, model, attack_db, pos_list,
            targeted_attack=False, physical_attack=False, parameters=None):

    if physical_attack:
        motion_degree = parameters.get("motion_degree", [0])
        motion_angle = parameters.get("motion_angle", [0])
        size_mul = parameters.get("size_mul", [1])
        brightness_mul = parameters.get("brightness_mul", [1])
        shadow_mul = parameters.get("shadow_mul", [0.5])
        shadow_move = parameters.get("shadow_move", [[0, 0]])
        perspective_mat = parameters.get("perspective_mat",
                                         [np.float32([[0, 0], [223, 0], [223, 223], [0, 223]])])
        img = image_transformation(img, position, pos_list, motion_degree, motion_angle,
                                   size_mul, brightness_mul, shadow_mul, shadow_move,
                                   perspective_mat, attack_db == "GTSRB").to(device)
    else:
        img, shadow_area = draw_shadow(position, img, pos_list, _shadow_level)
        img = shadow_edge_blur(img, shadow_area, 3)
        img = pre_process_image(img).astype(np.float32) if attack_db == "GTSRB" else img
        img = data_transforms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        predict = torch.softmax(model(img), 1)
        predict = torch.mean(predict, dim=0)
        label = parameters.get("label")

    if targeted_attack:
        target = parameters.get("target")
        confidence = float(1 - predict[target])
        success = torch.argmax(predict) == target
    else:
        confidence = float(predict[label])
        success = torch.argmax(predict) != label

    return confidence, success, predict


class Particle:

    # initialization
    def __init__(self, x_min, x_max, max_speed, dim, img, attack_db, pos_list, model,
                 targeted_attack, physical_attack, parameters):
        self.__pos = polygon_correction(np.array([random.uniform(x_min, x_max) for _ in range(dim)]))
        self.__speed = np.array([random.uniform(-max_speed, max_speed) for _ in range(dim)])
        self.__bestPos = np.array([0.0 for _ in range(dim)])
        self.__fitnessValue, _, _ = fit_fun(self.__pos, img, model, attack_db, pos_list,
                                            targeted_attack, physical_attack, parameters)

    def set_pos(self, i, value):

        self.__pos[i] = value

    def get_pos(self):

        return self.__pos

    def set_best_pos(self, i, value):

        self.__bestPos[i] = value

    def get_best_pos(self):

        return self.__bestPos

    def set_speed(self, i, value):

        self.__speed[i] = value

    def get_speed(self):

        return self.__speed

    def set_fitness_value(self, value):

        self.__fitnessValue = value

    def get_fitness_value(self):

        return self.__fitnessValue


class PSO:
    def __init__(self,
                 dim,        # Number of parameters to be optimized
                 size,       # Number of particles
                 iter_num,   # The maximum number of iterations
                 x_min,      # The minimum value of x and y (coord)
                 x_max,      # The maximum value of x and y (coord)
                 max_speed,  # The maximum speed of a particle
                 img,        # Our targeted image
                 attack_db,  # Our targeted dataset
                 pos_list,   # the area in mask M
                 model,      # Our targeted model
                 targeted_attack=False,
                 physical_attack=False,
                 parameters=None,
                 c1=2,
                 c2=2,
                 w=1):
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.dim = dim
        self.size = size
        self.iter_num = iter_num
        self.x_min = x_min
        self.x_max = x_max
        self.max_speed = max_speed
        self.img = img
        self.image_width = img.shape[1]
        self.image_height = img.shape[0]
        self.attack_db = attack_db
        self.pos_list = pos_list
        self.model = model
        self.best_fitness_value = float('Inf')
        self.best_position = [0.0 for _ in range(dim)]
        self.fitness_val_list = []
        self.success = False
        self.targeted_attack = targeted_attack
        self.physical_attack = physical_attack
        self.parameters = parameters

        global _shadow_level
        from shadow_attack import shadow_level
        _shadow_level = shadow_level

        # Population initialization
        self.Particle_list = [
            Particle(x_min, x_max, max_speed, dim, img, attack_db, pos_list, model,
                     targeted_attack, physical_attack, parameters) for _ in range(self.size)]

    def set_best_fitness_value(self, value):
        self.best_fitness_value = value

    def get_best_fitness_value(self):
        return self.best_fitness_value

    def set_best_position(self, i, value):
        self.best_position[i] = value

    def get_best_position(self):
        return self.best_position

    def update_speed(self, part):
        for i in range(self.dim):
            speed_value = self.w * part.get_speed()[i] \
                        + self.c1 * random.random() * (part.get_best_pos()[i] - part.get_pos()[i]) \
                        + self.c2 * random.random() * (self.get_best_position()[i] - part.get_pos()[i])
            if speed_value > self.max_speed:
                speed_value = self.max_speed
            elif speed_value < -self.max_speed:
                speed_value = -self.max_speed
            part.set_speed(i, speed_value)

    def update_pos(self, part, **parameters):
        speed = part.get_speed()
        position = polygon_correction(np.clip(part.get_pos(), self.x_min, self.x_max) + speed)
        for i in range(self.dim):
            part.set_pos(i, position[i])
        parameters.update(self.parameters)
        value, self.success, _ = fit_fun(position, self.img, self.model, self.attack_db, self.pos_list,
                                         self.targeted_attack, self.physical_attack, parameters)
        if value < part.get_fitness_value():
            part.set_fitness_value(value)
            for i in range(self.dim):
                part.set_best_pos(i, position[i])
        if value < self.get_best_fitness_value() or (self.success and not self.physical_attack):
            self.set_best_fitness_value(value)
            for i in range(self.dim):
                self.set_best_position(i, position[i])

    def update(self):
        i = j = 0
        for i in range(self.iter_num):
            j = 0
            for j, part in enumerate(self.Particle_list):
                self.update_speed(part)
                self.update_pos(part)
                if self.success:
                    break
            self.fitness_val_list.append(self.get_best_fitness_value())
            if self.success:
                break
        return self.fitness_val_list, self.get_best_position(), self.success, (i+1)*10+j+1

    def update_physical(self):
        num = self.parameters.get("transform_num")
        w, h = self.image_width, self.image_height

        for i in range(self.iter_num):
            print(f"iteration: {i + 1}", end=" ")
            for part in self.Particle_list:
                self.update_speed(part)
                _num = num if i < self.iter_num - 100 else 0
                self.update_pos(part, **random_param_generator(_num, w, h))
            self.fitness_val_list.append(self.get_best_fitness_value())
            print(self.get_best_fitness_value())

        return self.fitness_val_list, self.get_best_position(), self.success, -1
