import pygame
import random
import config as cfg
import time
import numpy as np
#import pandas as pd
import csv
import configparser


ext_cfg = configparser.ConfigParser()
ext_cfg.read('config.cfg')

pygame.init()

TRAINING_DATA_FILE = ext_cfg['Training']['training_data']
WIDTH = int(ext_cfg['Display']['width'])
HEIGHT = int(ext_cfg['Display']['height'])

NUM_EPOCHS = int(ext_cfg['Training']['epochs'])
LEARNING_RATE = float(ext_cfg['Training']['learning_rate'])

WHITE = cfg.WHITE
BLUE = cfg.BLUE
RED = cfg.RED
BRIGHT_RED = cfg.BRIGHT_RED
YELLOW = cfg.YELLOW
BLACK = cfg.BLACK
GREEN = cfg.GREEN
BRIGHT_GREEN = cfg.BRIGHT_GREEN
SILVER = cfg.SILVER
DARK_SILVER = cfg.DARK_SILVER
LIGHT_SILVER = cfg.LIGHT_SILVER


NETWORK_SPEED = 0.01

global NETWORK_ACTIVE
NETWORK_ACTIVE = False

global NETWORK_ACTIVATIONS
NETWORK_ACTIVATIONS = {}

global NETWORK_TENSORS
NETWORK_TENSORS = {}

global NEURONS
NEURONS = []

global INPUT_BOXES
INPUT_BOXES = []

global CURRENT_NUM_LAYERS
global CURRENT_NUM_NEURONS_INPUT
global CURRENT_NUM_NEURONS_HIDDEN
global CURRENT_NUM_NEURONS_OUTPUT
global INPUT_LAYER_INDEX
global OUTPUT_LAYER_INDEX

#training_data = pd.read_csv(cfg.TRAINING_DATA_FILE)

game_display = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Visual Network')
clock = pygame.time.Clock()

pygame.font.init()


def set_network_speed(speed):
    global NETWORK_SPEED
    NETWORK_SPEED = speed / cfg.MAX_SPEED


def get_network_speed():
    global NETWORK_SPEED
    return int(round(float(NETWORK_SPEED * cfg.MAX_SPEED), 0))


def convert_display_speed_to_delay(speed):
    return float(speed / cfg.MAX_SPEED)


def convert_delay_to_display_speed(delay):
    return int(delay * cfg.MAX_SPEED)


class Blob:
    def __init__(self, color):
        self.x = random.randrange(0, WIDTH)
        self.y = random.randrange(0, HEIGHT)
        self.size = random.randrange(4, 8)
        self.color = color

    def move(self):
        self.move_x = random.randrange(-1, 2)
        self.move_y = random.randrange(-1, 2)
        self.x += self.move_x
        self.y += self.move_y

        if self.x < 0:
            self.x = 0
        elif self.x > WIDTH:
            self.x = WIDTH

        if self.y < 0:
            self.y = 0
        elif self.y > HEIGHT:
            self.y = HEIGHT


class Neuron:
    def __init__(self, layer, node, activation, neuron_size, y_offset=0):
        self.layer = layer
        self.y_offset = y_offset
        self.x = cfg.NEURON_START_X + (layer * cfg.NEURON_SPACING_X)
        self.y = (cfg.NEURON_START_Y + (node * cfg.NEURON_SPACING_Y)) + self.y_offset
        self.size = neuron_size
        self.color = BLUE
        self.activation = activation
        self.node = node


def add_neuron(neuron):
    # text setting
    font = pygame.font.Font(cfg.NEURON_FONT, cfg.NEURON_FONT_SIZE)
    text_surface_obj = font.render(str(round(float(neuron.activation), 2)), True, BLACK, WHITE)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (neuron.x, neuron.y)

    pygame.draw.circle(game_display, neuron.color, [neuron.x, neuron.y], neuron.size)

    # Write weights onto neuron
    game_display.blit(text_surface_obj, text_rect_obj)

    pygame.display.update()


def update_neuron(neuron):
    # Text
    font = pygame.font.Font(cfg.NEURON_FONT, cfg.NEURON_FONT_SIZE)
    text_surface_obj = font.render(str(round(float(neuron.activation), 2)), True, BLACK, WHITE)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (neuron.x, neuron.y)
    pygame.draw.circle(game_display, RED, [neuron.x, neuron.y], neuron.size)

    # Write weights onto neuron
    game_display.blit(text_surface_obj, text_rect_obj)
    pygame.display.update()

    # Pause (flash)
    time.sleep(NETWORK_SPEED)

    pygame.draw.circle(game_display, neuron.color, [neuron.x, neuron.y], neuron.size)

    # Write weights onto neuron
    game_display.blit(text_surface_obj, text_rect_obj)
    pygame.display.update()

    # Pause
    time.sleep(NETWORK_SPEED)


def draw_controls():
    # Build network button
    refresh_build_network_button()

    # Start training button
    refresh_start_training_button()

    # Layers input box
    refresh_layers_input_box('3')
    refresh_increment_layers_tick_button()
    refresh_decrement_layers_tick_button()

    # Input layer input box
    refresh_input_layer_input_box(str(cfg.DEFAULT_NETWORK_INPUTS))
    refresh_increment_input_tick_button()
    refresh_decrement_input_tick_button()

    # Hidden layer input box
    refresh_hidden_layer_input_box(str(cfg.DEFAULT_NETWORK_HIDDEN))
    refresh_increment_hidden_tick_button()
    refresh_decrement_hidden_tick_button()

    # Output layer input box
    refresh_output_layer_input_box(str(cfg.DEFAULT_NETWORK_OUTPUTS))
    refresh_increment_output_tick_button()
    refresh_decrement_output_tick_button()

    # Message box bottom (output)
    refresh_message_box_top(text='Ready...')
    refresh_message_box_bottom(text='Click Build Network to begin...')

    # Loss
    refresh_text_box_loss(text='0.0')

    # Epochs
    refresh_text_box_epoch(text='0/' + str(NUM_EPOCHS))

    # Examples
    refresh_text_box_examples(text='0/0')

    # Speed
    set_network_speed(cfg.DEFAULT_NETWORK_SPEED)
    refresh_text_box_speed(text=str(convert_delay_to_display_speed(NETWORK_SPEED)))

    # Update screen
    pygame.display.update()


def build_network_layer(layer, num_neurons):
    # Build layer neurons
    global CURRENT_NUM_LAYERS
    global INPUT_LAYER_INDEX
    global OUTPUT_LAYER_INDEX

    neuron_var = 0
    neuron_y_offset = 0
    if layer == 0:
        neuron_var = abs(int(CURRENT_NUM_NEURONS_HIDDEN) - int(CURRENT_NUM_NEURONS_INPUT))
        neuron_y_offset = (int(neuron_var) * (cfg.NEURON_SPACING_Y)) // 2
    elif layer == int(CURRENT_NUM_LAYERS) - 1:
        neuron_var = abs(int(CURRENT_NUM_NEURONS_HIDDEN) - int(CURRENT_NUM_NEURONS_OUTPUT))
        neuron_y_offset = (int(neuron_var) * (cfg.NEURON_SPACING_Y))//2

    neurons = []
    for i in range(num_neurons):
        if layer == 0:
            # Input layer
            #neuron = Neuron(layer, i, str(training_data['x' + str(i+1)][0]), cfg.NEURON_SIZE, y_offset=neuron_y_offset)
            neuron = Neuron(layer, i, str(get_training_example_feature_csv(example_index=0, col_index=i)), cfg.NEURON_SIZE,
                            y_offset=neuron_y_offset)
        elif layer == int(CURRENT_NUM_LAYERS) - 1:
            # Output layer
            #neuron = Neuron(layer, i, str(training_data['y'][0]), cfg.NEURON_SIZE, y_offset=neuron_y_offset)
            neuron = Neuron(layer, i, str(get_training_example_feature_csv(example_index=0, col_index=2)), cfg.NEURON_SIZE, y_offset=neuron_y_offset)
        else:
            # Hidden layer
            neuron = Neuron(layer, i, str(round(random.uniform(0, 1), 2)), cfg.NEURON_SIZE)

        neurons.append(neuron)
        add_neuron(neurons[i])

    global NETWORK_ACTIVATIONS
    NETWORK_ACTIVATIONS[layer] = neurons

    global NETWORK_ACTIVE
    NETWORK_ACTIVE = True


def build_network_tensor(layer, neuron_L_index, neuron_nL_index):
    global NETWORK_ACTIVATIONS
    global NETWORK_TENSORS

    tensors = NETWORK_TENSORS[layer]
    tensor = round(tensors[neuron_nL_index, neuron_L_index], 2)

    neurons_L = NETWORK_ACTIVATIONS[layer]
    neurons_nL = NETWORK_ACTIVATIONS[layer + 1]

    x1 = neurons_L[neuron_L_index].x
    y1 = neurons_L[neuron_L_index].y

    x2 = neurons_nL[neuron_nL_index].x
    y2 = neurons_nL[neuron_nL_index].y

    pygame.draw.line(game_display, DARK_SILVER, (x1+20, y1), (x2-20, y2))

    if layer != 0:
        offset = 0
    else:
        offset = 1

    # Draw weight text
    w_x = (x1 + ((x2 - x1)//2)) - (offset * (12 * neuron_L_index))
    w_y = (y1 - ((y1 - y2)//2)) + (offset * (12 * neuron_L_index))

    font = pygame.font.Font(cfg.NEURON_FONT, cfg.TENSOR_FONT_SIZE)
    text_surface_obj = font.render(str(tensor), True, BLACK, WHITE)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (w_x, w_y)
    game_display.blit(text_surface_obj, text_rect_obj)


def flash_network_tensor(layer, neuron_L_index, neuron_nL_index):
    global NETWORK_ACTIVATIONS
    global NETWORK_TENSORS

    tensors = NETWORK_TENSORS[layer]
    tensor = round(tensors[neuron_nL_index, neuron_L_index], 2)

    neurons_L = NETWORK_ACTIVATIONS[layer]
    neurons_nL = NETWORK_ACTIVATIONS[layer + 1]

    x1 = neurons_L[neuron_L_index].x
    y1 = neurons_L[neuron_L_index].y

    x2 = neurons_nL[neuron_nL_index].x
    y2 = neurons_nL[neuron_nL_index].y

    pygame.draw.line(game_display, RED, (x1 + 20, y1), (x2 - 20, y2))

    if layer != 0:
        offset = 0
    else:
        offset = 1

    # Draw weight text
    w_x = (x1 + ((x2 - x1)//2)) - (offset * (12 * neuron_L_index))
    w_y = (y1 - ((y1 - y2)//2)) + (offset * (12 * neuron_L_index))

    font = pygame.font.Font(cfg.NEURON_FONT, cfg.TENSOR_FONT_SIZE)
    text_surface_obj = font.render(str(tensor), True, RED, YELLOW)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (w_x, w_y)
    game_display.blit(text_surface_obj, text_rect_obj)

    pygame.display.update()

    # Pause (flash)
    time.sleep(NETWORK_SPEED)

    pygame.draw.line(game_display, DARK_SILVER, (x1+20, y1), (x2-20, y2))

    if layer != 0:
        offset = 0
    else:
        offset = 1

    # Clear text
    font = pygame.font.Font(cfg.NEURON_FONT, cfg.TENSOR_FONT_SIZE)
    text_surface_obj = font.render((8 * ' '), True, BLACK, BLACK)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (w_x, w_y)
    game_display.blit(text_surface_obj, text_rect_obj)

    # Draw weight text
    w_x = (x1 + ((x2 - x1)//2)) - (offset * (12 * neuron_L_index))
    w_y = (y1 - ((y1 - y2)//2)) + (offset * (12 * neuron_L_index))

    font = pygame.font.Font(cfg.NEURON_FONT, cfg.TENSOR_FONT_SIZE)
    text_surface_obj = font.render(str(tensor), True, BLACK, WHITE)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (w_x, w_y)
    game_display.blit(text_surface_obj, text_rect_obj)


def build_network():
    global CURRENT_NUM_LAYERS
    global CURRENT_NUM_NEURONS_INPUT
    global CURRENT_NUM_NEURONS_HIDDEN
    global CURRENT_NUM_NEURONS_OUTPUT
    global INPUT_LAYER_INDEX
    global OUTPUT_LAYER_INDEX
    global NETWORK_TENSORS
    global NETWORK_ACTIVATIONS

    # Create neurons
    for i in range(int(CURRENT_NUM_LAYERS)):
        if i == 0:
            build_network_layer(i, int(CURRENT_NUM_NEURONS_INPUT))
        elif i == int(CURRENT_NUM_LAYERS) - 1:
            build_network_layer(i, int(CURRENT_NUM_NEURONS_OUTPUT))
        else:
            build_network_layer(i, int(CURRENT_NUM_NEURONS_HIDDEN))

    nn_initialize_tensors()

    # Create tensors
    # For each layer of tensors
    for i in range(len(NETWORK_TENSORS)):
        # Loop over layer L
        for iL in range(len(NETWORK_ACTIVATIONS[i])):
            # Loop over layer L + 1
            for iNL in range(len(NETWORK_ACTIVATIONS[i + 1])):
                build_network_tensor(i, iL, iNL)

    INPUT_LAYER_INDEX = 0
    OUTPUT_LAYER_INDEX = int(CURRENT_NUM_LAYERS) - 1


def refresh_build_network_button():
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    button_color = GREEN
    button_click = False

    # Check build network
    if cfg.BUTTON_BUILD_NETWORK_X + cfg.BUTTON_WIDTH > mouse[
        0] > cfg.BUTTON_BUILD_NETWORK_X and cfg.BUTTON_BUILD_NETWORK_Y + cfg.BUTTON_HEIGHT > mouse[
        1] > cfg.BUTTON_BUILD_NETWORK_Y:
        button_color = BRIGHT_GREEN

        if click[0] == 1:
            button_click = True
    else:
        button_color = GREEN

    # Draw button
    pygame.draw.rect(game_display, button_color,
                     (cfg.BUTTON_BUILD_NETWORK_X, cfg.BUTTON_BUILD_NETWORK_Y, cfg.BUTTON_WIDTH, cfg.BUTTON_HEIGHT))

    # Draw text
    font = pygame.font.Font(cfg.NEURON_FONT, cfg.NEURON_FONT_SIZE)
    text_surface_obj = font.render('Build Network', True, BLACK, button_color)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (
    cfg.BUTTON_BUILD_NETWORK_X + (cfg.BUTTON_WIDTH // 2), cfg.BUTTON_BUILD_NETWORK_Y + (cfg.BUTTON_HEIGHT // 2))

    game_display.blit(text_surface_obj, text_rect_obj)

    if button_click:
        build_network()


def refresh_start_training_button():
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    x = cfg.BUTTON_START_TRAINING_X
    y = cfg.BUTTON_START_TRAINING_Y
    w = cfg.BUTTON_WIDTH
    h = cfg.BUTTON_HEIGHT

    button_color = RED
    button_click = False

    # Check build network
    if x + w > mouse[0] > x and y + h > mouse[1] > y:
        button_color = BRIGHT_RED

        if click[0] == 1:
            button_click = True
    else:
        button_color = RED

    # Draw button
    pygame.draw.rect(game_display, button_color, (x, y, w, h))

    # Draw text
    font = pygame.font.Font(cfg.BUTTON_FONT, cfg.BUTTON_FONT_SIZE)
    text_surface_obj = font.render('Start Training', True, BLACK, button_color)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x + (w // 2), y + (h // 2))

    game_display.blit(text_surface_obj, text_rect_obj)

    if button_click:
        start_training()


def refresh_layers_input_box(text=''):
    x = cfg.INPUT_BOX_LAYERS_X
    y = cfg.INPUT_BOX_LAYERS_Y
    h = cfg.INPUT_BOX_LAYERS_HEIGHT
    w = cfg.INPUT_BOX_LAYERS_WIDTH
    font_size = cfg.INPUT_BOX_FONT_SIZE

    # Draw label
    smallText = pygame.font.SysFont(None, font_size)
    textSurf = smallText.render('Layers:', True, WHITE, BLACK)
    textRect = textSurf.get_rect()
    textRect.center = ((x + (w)) - int(font_size * 2.25), (y + (h / 2)))
    game_display.blit(textSurf, textRect)

    # Draw text
    font = pygame.font.Font(cfg.BUTTON_FONT, font_size)
    text_surface_obj = font.render(text, True, WHITE, BLACK)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x + (w // 2), y + (h // 2))

    game_display.blit(text_surface_obj, text_rect_obj)

    global CURRENT_NUM_LAYERS
    CURRENT_NUM_LAYERS = text


def refresh_input_layer_input_box(text=''):
    x = cfg.INPUT_BOX_INPUT_LAYER_X
    y = cfg.INPUT_BOX_INPUT_LAYER_Y
    h = cfg.INPUT_BOX_INPUT_LAYER_HEIGHT
    w = cfg.INPUT_BOX_INPUT_LAYER_WIDTH
    font_size = cfg.INPUT_BOX_FONT_SIZE

    # Draw label
    smallText = pygame.font.SysFont(None, font_size)
    textSurf = smallText.render('Input:', True, WHITE, BLACK)
    textRect = textSurf.get_rect()
    textRect.center = ((x + (w)) - int(font_size * 2.25), (y + (h / 2)))
    game_display.blit(textSurf, textRect)

    # Draw text
    font = pygame.font.Font(cfg.BUTTON_FONT, font_size)
    text_surface_obj = font.render(text, True, WHITE, BLACK)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x + (w // 2), y + (h // 2))

    game_display.blit(text_surface_obj, text_rect_obj)

    global CURRENT_NUM_NEURONS_INPUT
    CURRENT_NUM_NEURONS_INPUT = text


def refresh_hidden_layer_input_box(text=''):
    x = cfg.INPUT_BOX_HIDDEN_LAYER_X
    y = cfg.INPUT_BOX_HIDDEN_LAYER_Y
    h = cfg.INPUT_BOX_HIDDEN_LAYER_HEIGHT
    w = cfg.INPUT_BOX_HIDDEN_LAYER_WIDTH
    font_size = cfg.INPUT_BOX_FONT_SIZE

    # Draw label
    smallText = pygame.font.SysFont(None, font_size)
    textSurf = smallText.render('Hidden:', True, WHITE, BLACK)
    textRect = textSurf.get_rect()
    textRect.center = ((x + (w)) - int(font_size * 2.5), (y + (h / 2)))
    game_display.blit(textSurf, textRect)

    # Draw text
    font = pygame.font.Font(cfg.BUTTON_FONT, font_size)
    text_surface_obj = font.render(text, True, WHITE, BLACK)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x + (w // 2), y + (h // 2))

    game_display.blit(text_surface_obj, text_rect_obj)

    global CURRENT_NUM_NEURONS_HIDDEN
    CURRENT_NUM_NEURONS_HIDDEN = text


def refresh_output_layer_input_box(text=''):
    x = cfg.INPUT_BOX_OUTPUT_LAYER_X
    y = cfg.INPUT_BOX_OUTPUT_LAYER_Y
    h = cfg.INPUT_BOX_OUTPUT_LAYER_HEIGHT
    w = cfg.INPUT_BOX_OUTPUT_LAYER_WIDTH
    font_size = cfg.INPUT_BOX_FONT_SIZE

    # Draw label
    smallText = pygame.font.SysFont(None, font_size)
    textSurf = smallText.render('Output:', True, WHITE, BLACK)
    textRect = textSurf.get_rect()
    textRect.center = ((x + (w)) - int(font_size * 2.5), (y + (h / 2)))
    game_display.blit(textSurf, textRect)

    # Draw text
    font = pygame.font.Font(cfg.BUTTON_FONT, font_size)
    text_surface_obj = font.render(text, True, WHITE, BLACK)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x + (w // 2), y + (h // 2))

    game_display.blit(text_surface_obj, text_rect_obj)

    global CURRENT_NUM_NEURONS_OUTPUT
    CURRENT_NUM_NEURONS_OUTPUT = text


def refresh_increment_layers_tick_button():
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    button_color = SILVER
    button_click = False

    x = cfg.BUTTON_TICK_UP_1_X
    y = cfg.BUTTON_TICK_UP_1_Y
    w = cfg.BUTTON_TICK_WIDTH
    h = cfg.BUTTON_TICK_HEIGHT

    # Check build network
    if x + w > mouse[0] > x and y + h > mouse[1] > y:
        button_color = LIGHT_SILVER

        if click[0] == 1:
            button_click = True
    else:
        button_color = SILVER

    # Draw button
    pygame.draw.rect(game_display, button_color, (x, y, w, h))

    # Draw text
    font = pygame.font.Font(cfg.NEURON_FONT, cfg.NEURON_FONT_SIZE)
    text_surface_obj = font.render('+', True, BLACK, button_color)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x + (w // 2), y + (h // 2))

    game_display.blit(text_surface_obj, text_rect_obj)

    if button_click:
        increment_layers()


def refresh_decrement_layers_tick_button():
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    button_color = SILVER
    button_click = False

    x = cfg.BUTTON_TICK_DOWN_1_X
    y = cfg.BUTTON_TICK_DOWN_1_Y
    w = cfg.BUTTON_TICK_WIDTH
    h = cfg.BUTTON_TICK_HEIGHT

    # Check build network
    if x + w > mouse[0] > x and y + h > mouse[1] > y:
        button_color = LIGHT_SILVER

        if click[0] == 1:
            button_click = True
    else:
        button_color = SILVER

    # Draw button
    pygame.draw.rect(game_display, button_color, (x, y, w, h))

    # Draw text
    font = pygame.font.Font(cfg.NEURON_FONT, cfg.NEURON_FONT_SIZE)
    text_surface_obj = font.render('-', True, BLACK, button_color)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x + (w // 2), y + (h // 2))

    game_display.blit(text_surface_obj, text_rect_obj)

    if button_click:
        decrement_layers()


def refresh_increment_input_tick_button():
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    button_color = SILVER
    button_click = False

    x = cfg.BUTTON_TICK_UP_INPUT_X
    y = cfg.BUTTON_TICK_UP_INPUT_Y
    w = cfg.BUTTON_TICK_WIDTH
    h = cfg.BUTTON_TICK_HEIGHT

    # Check build network
    if x + w > mouse[0] > x and y + h > mouse[1] > y:
        button_color = LIGHT_SILVER

        if click[0] == 1:
            button_click = True
    else:
        button_color = SILVER

    # Draw button
    pygame.draw.rect(game_display, button_color, (x, y, w, h))

    # Draw text
    font = pygame.font.Font(cfg.NEURON_FONT, cfg.NEURON_FONT_SIZE)
    text_surface_obj = font.render('+', True, BLACK, button_color)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x + (w // 2), y + (h // 2))

    game_display.blit(text_surface_obj, text_rect_obj)

    if button_click:
        increment_input()


def refresh_decrement_input_tick_button():
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    button_color = SILVER
    button_click = False

    x = cfg.BUTTON_TICK_DOWN_INPUT_X
    y = cfg.BUTTON_TICK_DOWN_INPUT_Y
    w = cfg.BUTTON_TICK_WIDTH
    h = cfg.BUTTON_TICK_HEIGHT

    # Check build network
    if x + w > mouse[0] > x and y + h > mouse[1] > y:
        button_color = LIGHT_SILVER

        if click[0] == 1:
            button_click = True
    else:
        button_color = SILVER

    # Draw button
    pygame.draw.rect(game_display, button_color, (x, y, w, h))

    # Draw text
    font = pygame.font.Font(cfg.NEURON_FONT, cfg.NEURON_FONT_SIZE)
    text_surface_obj = font.render('-', True, BLACK, button_color)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x + (w // 2), y + (h // 2))

    game_display.blit(text_surface_obj, text_rect_obj)

    if button_click:
        decrement_input()


def refresh_increment_hidden_tick_button():
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    button_color = SILVER
    button_click = False

    x = cfg.BUTTON_TICK_UP_HIDDEN_X
    y = cfg.BUTTON_TICK_UP_HIDDEN_Y
    w = cfg.BUTTON_TICK_WIDTH
    h = cfg.BUTTON_TICK_HEIGHT

    # Check build network
    if x + w > mouse[0] > x and y + h > mouse[1] > y:
        button_color = LIGHT_SILVER

        if click[0] == 1:
            button_click = True
    else:
        button_color = SILVER

    # Draw button
    pygame.draw.rect(game_display, button_color, (x, y, w, h))

    # Draw text
    font = pygame.font.Font(cfg.NEURON_FONT, cfg.NEURON_FONT_SIZE)
    text_surface_obj = font.render('+', True, BLACK, button_color)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x + (w // 2), y + (h // 2))

    game_display.blit(text_surface_obj, text_rect_obj)

    if button_click:
        increment_hidden()


def refresh_decrement_hidden_tick_button():
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    button_color = SILVER
    button_click = False

    x = cfg.BUTTON_TICK_DOWN_HIDDEN_X
    y = cfg.BUTTON_TICK_DOWN_HIDDEN_Y
    w = cfg.BUTTON_TICK_WIDTH
    h = cfg.BUTTON_TICK_HEIGHT

    # Check build network
    if x + w > mouse[0] > x and y + h > mouse[1] > y:
        button_color = LIGHT_SILVER

        if click[0] == 1:
            button_click = True
    else:
        button_color = SILVER

    # Draw button
    pygame.draw.rect(game_display, button_color, (x, y, w, h))

    # Draw text
    font = pygame.font.Font(cfg.NEURON_FONT, cfg.NEURON_FONT_SIZE)
    text_surface_obj = font.render('-', True, BLACK, button_color)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x + (w // 2), y + (h // 2))

    game_display.blit(text_surface_obj, text_rect_obj)

    if button_click:
        decrement_hidden()


def refresh_increment_output_tick_button():
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    button_color = SILVER
    button_click = False

    x = cfg.BUTTON_TICK_UP_OUTPUT_X
    y = cfg.BUTTON_TICK_UP_OUTPUT_Y
    w = cfg.BUTTON_TICK_WIDTH
    h = cfg.BUTTON_TICK_HEIGHT

    # Check build network
    if x + w > mouse[0] > x and y + h > mouse[1] > y:
        button_color = LIGHT_SILVER

        if click[0] == 1:
            button_click = True
    else:
        button_color = SILVER

    # Draw button
    pygame.draw.rect(game_display, button_color, (x, y, w, h))

    # Draw text
    font = pygame.font.Font(cfg.NEURON_FONT, cfg.NEURON_FONT_SIZE)
    text_surface_obj = font.render('+', True, BLACK, button_color)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x + (w // 2), y + (h // 2))

    game_display.blit(text_surface_obj, text_rect_obj)

    if button_click:
        increment_output()


def refresh_decrement_output_tick_button():
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    button_color = SILVER
    button_click = False

    x = cfg.BUTTON_TICK_DOWN_OUTPUT_X
    y = cfg.BUTTON_TICK_DOWN_OUTPUT_Y
    w = cfg.BUTTON_TICK_WIDTH
    h = cfg.BUTTON_TICK_HEIGHT

    # Check build network
    if x + w > mouse[0] > x and y + h > mouse[1] > y:
        button_color = LIGHT_SILVER

        if click[0] == 1:
            button_click = True
    else:
        button_color = SILVER

    # Draw button
    pygame.draw.rect(game_display, button_color, (x, y, w, h))

    # Draw text
    font = pygame.font.Font(cfg.NEURON_FONT, cfg.NEURON_FONT_SIZE)
    text_surface_obj = font.render('-', True, BLACK, button_color)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x + (w // 2), y + (h // 2))

    game_display.blit(text_surface_obj, text_rect_obj)

    if button_click:
        decrement_output()


def increment_layers():
    global CURRENT_NUM_LAYERS
    num_layers = int(CURRENT_NUM_LAYERS) + 1
    if num_layers > cfg.MAX_LAYERS:
        num_layers = int(cfg.MAX_LAYERS)

    refresh_layers_input_box(str(num_layers))


def decrement_layers():
    global CURRENT_NUM_LAYERS
    num_layers = int(CURRENT_NUM_LAYERS) - 1
    if num_layers < cfg.MIN_LAYERS:
        num_layers = int(cfg.MIN_LAYERS)

    refresh_layers_input_box(str(num_layers))


def increment_input():
    global CURRENT_NUM_NEURONS_INPUT
    new_num = int(CURRENT_NUM_NEURONS_INPUT) + 1
    if new_num > cfg.MAX_NEURONS:
        new_num = int(cfg.MAX_NEURONS)

    refresh_input_layer_input_box(str(new_num))


def decrement_input():
    global CURRENT_NUM_NEURONS_INPUT
    new_num = int(CURRENT_NUM_NEURONS_INPUT) - 1
    if new_num < cfg.MIN_NEURONS:
        new_num = int(cfg.MIN_NEURONS)

    refresh_input_layer_input_box(str(new_num))


def increment_hidden():
    global CURRENT_NUM_NEURONS_HIDDEN
    new_num = int(CURRENT_NUM_NEURONS_HIDDEN) + 1
    if new_num > cfg.MAX_NEURONS:
        new_num = int(cfg.MAX_NEURONS)

    refresh_hidden_layer_input_box(str(new_num))


def decrement_hidden():
    global CURRENT_NUM_NEURONS_HIDDEN
    new_num = int(CURRENT_NUM_NEURONS_HIDDEN) - 1
    if new_num < cfg.MIN_NEURONS:
        new_num = int(cfg.MIN_NEURONS)

    refresh_hidden_layer_input_box(str(new_num))


def increment_output():
    global CURRENT_NUM_NEURONS_OUTPUT
    new_num = int(CURRENT_NUM_NEURONS_OUTPUT) + 1
    if new_num > cfg.MAX_NEURONS:
        new_num = int(cfg.MAX_NEURONS)

    refresh_output_layer_input_box(str(new_num))


def decrement_output():
    global CURRENT_NUM_NEURONS_OUTPUT
    new_num = int(CURRENT_NUM_NEURONS_OUTPUT) - 1
    if new_num < cfg.MIN_NEURONS:
        new_num = int(cfg.MIN_NEURONS)

    refresh_output_layer_input_box(str(new_num))


def refresh_message_box_bottom(text=''):
    x = cfg.MESSAGE_BOX_BOTTOM_X
    y = cfg.MESSAGE_BOX_BOTTOM_Y
    h = cfg.MESSAGE_BOX_BOTTOM_HEIGHT
    w = cfg.MESSAGE_BOX_BOTTOM_WIDTH
    font_size = cfg.MESSAGE_BOX_FONT_SIZE

    # Draw label
    smallText = pygame.font.SysFont(None, font_size)
    textSurf = smallText.render('', True, WHITE, BLACK)
    textRect = textSurf.get_rect()
    textRect.center = (x - 50, (y + (h / 2)))
    game_display.blit(textSurf, textRect)

    # Clear text
    font = pygame.font.Font(cfg.BUTTON_FONT, font_size)
    text_surface_obj = font.render((500 * ' '), True, WHITE, BLACK)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x, y)
    game_display.blit(text_surface_obj, text_rect_obj)

    # Draw text
    font = pygame.font.Font(cfg.BUTTON_FONT, font_size)
    text_surface_obj = font.render(text, True, WHITE, BLACK)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x, y)
    game_display.blit(text_surface_obj, text_rect_obj)
    pygame.display.update()
    time.sleep(NETWORK_SPEED)


def refresh_message_box_top(text=''):
    x = cfg.MESSAGE_BOX_TOP_X
    y = cfg.MESSAGE_BOX_TOP_Y
    h = cfg.MESSAGE_BOX_TOP_HEIGHT
    w = cfg.MESSAGE_BOX_TOP_WIDTH
    font_size = cfg.MESSAGE_BOX_FONT_SIZE

    # Draw label
    smallText = pygame.font.SysFont(None, font_size)
    textSurf = smallText.render('', True, WHITE, BLACK)
    textRect = textSurf.get_rect()
    textRect.center = (x - 50, (y + (h / 2)))
    game_display.blit(textSurf, textRect)

    # Clear text
    font = pygame.font.Font(cfg.BUTTON_FONT, font_size)
    text_surface_obj = font.render((500 * ' '), True, WHITE, BLACK)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x, y)
    game_display.blit(text_surface_obj, text_rect_obj)

    # Draw text
    font = pygame.font.Font(cfg.BUTTON_FONT, font_size)
    text_surface_obj = font.render(text, True, WHITE, BLACK)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x, y)
    game_display.blit(text_surface_obj, text_rect_obj)
    pygame.display.update()
    time.sleep(NETWORK_SPEED)


def refresh_text_box_loss(text=''):
    x = cfg.TEXT_BOX_LOSS_X
    y = cfg.TEXT_BOX_LOSS_Y
    h = cfg.TEXT_BOX_LOSS_HEIGHT
    w = cfg.TEXT_BOX_LOSS_WIDTH
    font_size = cfg.TEXT_BOX_LOSS_FONT_SIZE

    # Draw label
    smallText = pygame.font.SysFont(None, font_size)
    textSurf = smallText.render('Error(MSE):', True, WHITE, BLACK)
    textRect = textSurf.get_rect()
    textRect.center = ((x - (w)) - int(font_size * 2.5), (y))
    game_display.blit(textSurf, textRect)

    # Clear text
    font = pygame.font.Font(cfg.BUTTON_FONT, font_size)
    text_surface_obj = font.render((10 * ' '), True, WHITE, BLACK)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x, y)
    game_display.blit(text_surface_obj, text_rect_obj)

    # Draw text
    font = pygame.font.Font(cfg.BUTTON_FONT, font_size)
    text_surface_obj = font.render(text, True, WHITE, BLACK)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x, y)
    game_display.blit(text_surface_obj, text_rect_obj)
    pygame.display.update()


def refresh_text_box_epoch(text=''):
    x = cfg.TEXT_BOX_EPOCH_X
    y = cfg.TEXT_BOX_EPOCH_Y
    h = cfg.TEXT_BOX_EPOCH_HEIGHT
    w = cfg.TEXT_BOX_EPOCH_WIDTH
    font_size = cfg.TEXT_BOX_EPOCH_FONT_SIZE

    # Draw label
    smallText = pygame.font.SysFont(None, font_size)
    textSurf = smallText.render('Epoch:', True, WHITE, BLACK)
    textRect = textSurf.get_rect()
    textRect.center = ((x - (w)) - int(font_size * 1.5), (y))
    game_display.blit(textSurf, textRect)

    # Clear text
    font = pygame.font.Font(cfg.BUTTON_FONT, font_size)
    text_surface_obj = font.render((10 * ' '), True, WHITE, BLACK)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x, y)
    game_display.blit(text_surface_obj, text_rect_obj)

    # Draw text
    font = pygame.font.Font(cfg.BUTTON_FONT, font_size)
    text_surface_obj = font.render(text, True, WHITE, BLACK)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x, y)
    game_display.blit(text_surface_obj, text_rect_obj)
    pygame.display.update()


def refresh_text_box_examples(text=''):
    x = cfg.TEXT_BOX_EXAMPLES_X
    y = cfg.TEXT_BOX_EXAMPLES_Y
    h = cfg.TEXT_BOX_EXAMPLES_HEIGHT
    w = cfg.TEXT_BOX_EXAMPLES_WIDTH
    font_size = cfg.TEXT_BOX_EXAMPLES_FONT_SIZE

    # Draw label
    smallText = pygame.font.SysFont(None, font_size)
    textSurf = smallText.render('Example:', True, WHITE, BLACK)
    textRect = textSurf.get_rect()
    textRect.center = ((x - (w)) - int(font_size * 2), (y))
    game_display.blit(textSurf, textRect)

    # Clear text
    font = pygame.font.Font(cfg.BUTTON_FONT, font_size)
    text_surface_obj = font.render((10 * ' '), True, WHITE, BLACK)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x, y)
    game_display.blit(text_surface_obj, text_rect_obj)

    # Draw text
    font = pygame.font.Font(cfg.BUTTON_FONT, font_size)
    text_surface_obj = font.render(text, True, WHITE, BLACK)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x, y)
    game_display.blit(text_surface_obj, text_rect_obj)
    pygame.display.update()


def refresh_text_box_speed(text=''):
    x = cfg.TEXT_BOX_SPEED_X
    y = cfg.TEXT_BOX_SPEED_Y
    h = cfg.TEXT_BOX_SPEED_HEIGHT
    w = cfg.TEXT_BOX_SPEED_WIDTH
    font_size = cfg.TEXT_BOX_SPEED_FONT_SIZE

    text = str(100 - int(text))

    # Draw label
    smallText = pygame.font.SysFont(None, font_size)
    textSurf = smallText.render('Speed:', True, WHITE, BLACK)
    textRect = textSurf.get_rect()
    textRect.center = ((x - (w)) - int(font_size * 2), (y))
    game_display.blit(textSurf, textRect)

    # Clear text
    font = pygame.font.Font(cfg.BUTTON_FONT, font_size)
    text_surface_obj = font.render((8 * ' '), True, BLACK, BLACK)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x, y)
    game_display.blit(text_surface_obj, text_rect_obj)

    # Draw text
    font = pygame.font.Font(cfg.BUTTON_FONT, font_size)
    text_surface_obj = font.render(text, True, WHITE, BLACK)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x, y)
    game_display.blit(text_surface_obj, text_rect_obj)
    pygame.display.update()


def refresh_increment_speed_tick_button():
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    button_color = SILVER
    button_click = False

    x = cfg.BUTTON_TICK_UP_SPEED_X
    y = cfg.BUTTON_TICK_UP_SPEED_Y
    w = cfg.BUTTON_TICK_WIDTH
    h = cfg.BUTTON_TICK_HEIGHT

    # Check build network
    if x + w > mouse[0] > x and y + h > mouse[1] > y:
        button_color = LIGHT_SILVER

        if click[0] == 1:
            button_click = True
    else:
        button_color = SILVER

    # Draw button
    pygame.draw.rect(game_display, button_color, (x, y, w, h))

    # Draw text
    font = pygame.font.Font(cfg.NEURON_FONT, cfg.NEURON_FONT_SIZE)
    text_surface_obj = font.render('+', True, BLACK, button_color)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x + (w // 2), y + (h // 2))

    game_display.blit(text_surface_obj, text_rect_obj)

    if button_click:
        decrement_speed()


def refresh_decrement_speed_tick_button():
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    button_color = SILVER
    button_click = False

    x = cfg.BUTTON_TICK_DOWN_SPEED_X
    y = cfg.BUTTON_TICK_DOWN_SPEED_Y
    w = cfg.BUTTON_TICK_WIDTH
    h = cfg.BUTTON_TICK_HEIGHT

    # Check build network
    if x + w > mouse[0] > x and y + h > mouse[1] > y:
        button_color = LIGHT_SILVER

        if click[0] == 1:
            button_click = True
    else:
        button_color = SILVER

    # Draw button
    pygame.draw.rect(game_display, button_color, (x, y, w, h))

    # Draw text
    font = pygame.font.Font(cfg.NEURON_FONT, cfg.NEURON_FONT_SIZE)
    text_surface_obj = font.render('-', True, BLACK, button_color)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = (x + (w // 2), y + (h // 2))

    game_display.blit(text_surface_obj, text_rect_obj)

    if button_click:
        increment_speed()


def increment_speed():
    current_speed = get_network_speed()
    new_speed = current_speed + 1

    if new_speed > cfg.MAX_SPEED:
        new_speed = cfg.MAX_SPEED

    set_network_speed(new_speed)
    refresh_text_box_speed(str(new_speed))


def decrement_speed():
    current_speed = get_network_speed()
    new_speed = current_speed - 1
    if new_speed < 0:
        new_speed = 0

    set_network_speed(new_speed)
    refresh_text_box_speed(str(new_speed))


def draw_button(msg, x, y, w, h, ic, ac, action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    print(click)
    if x+w > mouse[0] > x and y+h > mouse[1] > y:
        pygame.draw.rect(game_display, ac, (x, y, w, h))

        if click[0] == 1 and action != None:
            action()
    else:
        pygame.draw.rect(game_display, ic, (x, y, w, h))

    smallText = pygame.font.SysFont(cfg.BUTTON_FONT, cfg.BUTTON_FONT_SIZE)
    textSurf = smallText.render(msg, True, BLACK, GREEN)
    textRect = textSurf.get_rect()
    textRect.center = ((x+(w/2)), (y+(h/2)))
    game_display.blit(textSurf, textRect)


class InputBox:
    def __init__(self, x, y, w, h, text='', label=''):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = SILVER
        self.text = text
        self.FONT = pygame.font.Font(None, 32)
        self.txt_surface = self.FONT.render(text, True, self.color)
        self.active = False
        self.COLOR_ACTIVE = LIGHT_SILVER
        self.COLOR_INACTIVE = SILVER
        self.label = label
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONUP:
            # If the user clicked on the input_box rect.
            if self.rect.collidepoint(event.pos):
                # Toggle the active variable.
                self.active = not self.active
            else:
                self.active = False
            # Change the current color of the input box.
            self.color = self.COLOR_ACTIVE if self.active else self.COLOR_INACTIVE
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    print(self.text)
                    self.text = ''
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                # Re-render the text
                    self.txt_surface = self.FONT.render(self.text, True, self.color)

    def update(self):
        # Resize the box if the text is too long.
        width = max(30, self.txt_surface.get_width()+10)
        self.rect.w = width

    def update_text(self):
        self.txt_surface = self.FONT.render(self.text, True, self.color)

    def draw_label(self, text):
        smallText = pygame.font.SysFont(cfg.BUTTON_FONT, cfg.BUTTON_FONT_SIZE)
        textSurf = smallText.render(text, True, BLACK, WHITE)
        textRect = textSurf.get_rect()
        textRect.center = ((self.x + (self.w))-(len(text) * 8), (self.y + (self.h / 2)))
        game_display.blit(textSurf, textRect)

    def draw(self, screen):
        # Blit the text.
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
        # Blit the rect.
        pygame.draw.rect(game_display, self.color, self.rect, 2)
        self.draw_label(text=self.label)


def check_controls():
    refresh_build_network_button()
    refresh_start_training_button()
    refresh_increment_layers_tick_button()
    refresh_decrement_layers_tick_button()
    refresh_increment_input_tick_button()
    refresh_decrement_input_tick_button()
    refresh_increment_hidden_tick_button()
    refresh_decrement_hidden_tick_button()
    refresh_increment_output_tick_button()
    refresh_decrement_output_tick_button()
    refresh_increment_speed_tick_button()
    refresh_decrement_speed_tick_button()
    pygame.display.update()


def nn_initialize_tensors():
    global NETWORK_TENSORS
    global NETWORK_ACTIVATIONS

    layer_tensors = []
    for i in range(int(CURRENT_NUM_LAYERS)-1):
        num_inputs = len(NETWORK_ACTIVATIONS[i])
        num_outputs = len(NETWORK_ACTIVATIONS[i + 1])

        # Randomly initialize hidden layer weights
        layer_tensors.append(np.random.rand(num_outputs, num_inputs))
        # bias = np.zeros((num_outputs, 1))

        NETWORK_TENSORS[i] = layer_tensors[i]


def sigmoid(z):
    return (1 / (1 + np.exp(-z)))


def load_training_example(example_index):
    global NETWORK_ACTIVATIONS
    input_layer = NETWORK_ACTIVATIONS[0]
    i = 0
    for neuron in input_layer:
        #temp_example = training_data['x' + str(i + 1)][example_index]
        temp_example = get_training_example_feature_csv(example_index=example_index, col_index=i)
        neuron.activation = temp_example
        input_layer[i] = neuron
        neuron.activation = temp_example
        update_neuron(neuron)
        i += 1


def get_training_inputs(example_index):
    global NETWORK_ACTIVATIONS
    input_layer = NETWORK_ACTIVATIONS[0]
    input_data = []
    i = 0
    for neuron in NETWORK_ACTIVATIONS[0]:
        #temp_example = training_data['x' + str(i + 1)][example_index]
        temp_example = get_training_example_feature_csv(example_index=example_index, col_index=i)
        input_data.append(temp_example)
        neuron.activation = temp_example
        update_neuron(neuron)
        i += 1

    return input_data


def get_training_example_feature_csv(example_index, col_index):
    example_index += 1  # Skip header row
    with open(TRAINING_DATA_FILE, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in csv_reader:
            if i == example_index:
                return float(row[col_index])
            else:
                i += 1


def get_training_example_label_csv(example_index):
    example_index += 1  # Skip header row
    with open(TRAINING_DATA_FILE, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in csv_reader:
            if i == example_index:
                return float(row[2])
            else:
                i += 1


def get_num_training_examples():
    with open(TRAINING_DATA_FILE, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        i = 0
        for _ in csv_reader:
                i += 1
    return i-1


def start_training():
    global CURRENT_NUM_LAYERS
    global NETWORK_ACTIVATIONS
    global NETWORK_ACTIVE

    if NETWORK_ACTIVE:
        #num_training_examples = len(training_data['y'])
        num_training_examples = get_num_training_examples()
        for iE in range(int(NUM_EPOCHS)):
            # Epochs
            refresh_text_box_epoch(text=str(iE + 1) + '/' + str(NUM_EPOCHS))

            print(86 * '=')
            print('Processing network...')
            print('Total training examples:', num_training_examples)
            # For each epoch...
            a_cache = {}
            z_cache = {}
            total_loss = 0
            refresh_message_box_top(text='Forward Propagation >>>>>>>')
            for iT in range(num_training_examples):  # For each training example
                # Examples
                refresh_text_box_examples(text=str(iT + 1) + '/' + str(num_training_examples))

                a_cache_examples = {}
                z_cache_examples = {}
                load_training_example(iT)

                #y = training_data['y'][iT]
                y = get_training_example_label_csv(iT)
                for iL in range(int(CURRENT_NUM_LAYERS)):
                    z_cache_temp = []
                    a_cache_temp = []
                    if iL > INPUT_LAYER_INDEX:
                        tensors = NETWORK_TENSORS[iL - 1]
                        i = 0
                        for neuron in NETWORK_ACTIVATIONS[iL]:
                            # Calculate activations for each neuron in hidden layers
                            print('Hidden Node Layer', neuron.layer)
                            print('Hidden Node #', neuron.node)
                            print('Tensor', tensors[i])
                            # Get inputs/activations from previous layer (multiply each neuron * weight)
                            pl_i = 0
                            z = 0
                            for prev_layer_neuron in NETWORK_ACTIVATIONS[iL - 1]:
                                z += (float(prev_layer_neuron.activation) * tensors[i][pl_i])
                                print('z += ', float(prev_layer_neuron.activation), ' * ', tensors[i][pl_i])

                                text = 'Z += (' + str(round(float(prev_layer_neuron.activation), 2)) + ' * ' + str(
                                    round(tensors[i][pl_i], 2)) + ') = ' + str(round((float(prev_layer_neuron.activation) * tensors[i][pl_i]), 2))
                                refresh_message_box_bottom(text=text)

                                flash_network_tensor(iL - 1, prev_layer_neuron.node, neuron.node)

                                pl_i += 1

                            text = 'Z = ' + str(round(z, 2))
                            refresh_message_box_bottom(text=text)
                            time.sleep(NETWORK_SPEED)
                            time.sleep(NETWORK_SPEED)

                            # Compute activation
                            a = sigmoid(z)
                            text = 'a = sigmoid(' + str(round(z, 2)) + ') = ' + str(round(a, 2))
                            refresh_message_box_bottom(text=text)

                            # Cache values for back prop
                            z_cache_temp.append(z)
                            a_cache_temp.append(a)

                            print('Z = ', z)
                            print('A = ', a)
                            neuron.activation = str(a)
                            update_neuron(neuron)
                            i += 1

                        # Cache values for each layer
                        z_cache_examples[iL - 1] = z_cache_temp
                        a_cache_examples[iL - 1] = a_cache_temp

                    if iL == OUTPUT_LAYER_INDEX:
                        print(86 * '=')
                        print('Example:', iT + 1, 'feed forward complete, calculating loss...')
                        y_hat = float(NETWORK_ACTIVATIONS[iL][0].activation)
                        loss = (int(y) - y_hat)**2
                        total_loss += loss

                        #refresh_text_box_loss(str(round(total_loss, 2)))

                        text = 'Error/Loss = (y - y^)^2 = (' + str(int(y)) + ' - ' + str(round(y_hat, 2)) + ')^2 = ' + str(round(total_loss, 2))
                        refresh_message_box_bottom(text=text)

                # Add cache values for each example (every example, every layer, every activation)
                z_cache[iT] = z_cache_examples
                a_cache[iT] = a_cache_examples


            print('Forward propagation complete.')
            print(86 * '=')
            total_loss_before = total_loss
            total_loss = (1 / num_training_examples) * total_loss

            refresh_message_box_bottom(text='MSE = 1/' + str(num_training_examples) + '(' + str(round(total_loss_before, 2)) + ') = ' + str(round(total_loss, 2)))
            refresh_text_box_loss(str(round(total_loss, 2)))

            print('Starting back propagation...')
            refresh_message_box_top(text='Back Propagation <<<<<<<')

            # todo REMOVED CODE CHECK HERE!!!!!!!!!!
            #global NETWORK_TENSORS
            for iT in range(num_training_examples):  # For each training example
                # Examples
                refresh_text_box_examples(text=str(iT + 1) + '/' + str(num_training_examples))

                # Use index to loop over each layer
                layer_index = int(CURRENT_NUM_LAYERS) - 2

                # Get cached values for this example
                a_cache_example = a_cache[iT]
                z_cache_example = z_cache[iT]

                # Calculate output layer (single neuron binary output) get_training_example_label_csv
                #y = training_data['y'][iT]
                y = get_training_example_label_csv(iT)
                y_hat = a_cache_example[int(CURRENT_NUM_LAYERS) - 2][0]

                # 01 - Cost w/r output
                dZ_output = (y - y_hat)**2
                print('Cost w/r Activation:', dZ_output)

                text = 'Cost w/r output = (y_hat - y)^2 = (' + str(round(y_hat, 2)) + ' - ' + str(round(y, 2)) + ')^2 = ' + str(round(dZ_output, 2))
                refresh_message_box_bottom(text=text)

                # todo FOR EACH LAYER OF WEIGHTS!!!
                # Calculate gradients for each layer and neuron in remaining layers back
                dA = {}
                dW = {}
                for iL in range(layer_index + 1):  # For each SET of tensors/weights
                    activations = a_cache_example[layer_index]
                    if layer_index - 1 >= 0:
                        prev_activations = a_cache_example[layer_index - 1]
                    else:
                        # Use inputs
                        prev_activations = get_training_inputs(iT)
                    weights = NETWORK_TENSORS[layer_index]
                    print('layer_index', layer_index)
                    print('activations:', activations)
                    print('prev_activations:', prev_activations)
                    print('weights:', weights)

                    # 02 - Calculate activations w/r to Z (i.e. A' prime)
                    dA_temp = []
                    for activation in activations:  # For each activation
                        dA_temp.append((activation * (1 - activation)))

                        text = 'A w/r Z = A(1 - A) = ' + str(round(activation, 2)) + '(1 - ' + str(round(activation, 2)) + ') = ' + str(round((activation * (1 - activation)), 2))
                        refresh_message_box_bottom(text=text)

                    # Add to cache of activations w/r Z
                    dA[iL] = dA_temp

                    curr_weights = weights
                    # For each activation in L and each W/tensor attached to previous layer L-1
                    dW_temp = []
                    i_AL = 0
                    for AL in dA[iL]:
                        i = 0
                        # todo FIX - Only looping through first SET of tensors instead of each weight value
                        # todo i.e. only processing a single set of connections between layers
                        curr_layer_weights = weights[i_AL]
                        for iW in range(len(curr_layer_weights)):
                            #left_layer = iW
                            #right_layer = i_AL

                            flash_network_tensor(layer_index, iW, i_AL)

                            w = curr_layer_weights[iW]

                            # 03 - Change in Z w/r to W = a(L-1) or previous layer activation
                            p_A = prev_activations[iW]
                            text = 'Z w/r W = a(L-1) = ' + str(round(p_A, 2))
                            refresh_message_box_bottom(text=text)

                            if iL == 0:
                                # Output layer
                                CW = dZ_output * AL * p_A
                            else:
                                # All previous layers
                                CW = dZ_output * AL * p_A * w

                            dW_temp.append(CW)
                            i += 1

                            text = 'Cost w/r W' + str(i_AL) + str(iW) + ' = ' + str(round(CW, 2))
                            refresh_message_box_bottom(text=text)

                        i_AL += 1

                    dW[iL] = dW_temp

                    text = 'Back prop complete for layer ' + str(layer_index) + ' example ' + str(iT)
                    refresh_message_box_bottom(text=text)

                    # Decrement layer index as output layer was processed above
                    layer_index -= 1

                # Update weights for each example
                i_layer = 0
                for i_layer in range(len(dW)):   # for each layer
                    tensor_layer = (int(CURRENT_NUM_LAYERS) - 2) - i_layer
                    original_weights = np.array(NETWORK_TENSORS[tensor_layer])
                    weights = np.array(NETWORK_TENSORS[tensor_layer]).flatten()

                    for i_weight in range(len(dW[i_layer])):
                        dw = dW[i_layer][i_weight]
                        w = weights[i_weight]
                        new_w = w - LEARNING_RATE * dw
                        weights[i_weight] = new_w

                        value_indexes = np.where(original_weights == w)
                        value_row = value_indexes[0][0]
                        value_col = value_indexes[1][0]

                        original_weights[value_row, value_col] = new_w

                        build_network_tensor(tensor_layer, value_col, value_row)

                    NETWORK_TENSORS[tensor_layer] = original_weights

                text = 'Back prop complete for layer ' + str(layer_index + 1) + ' example ' + str(iT + 1)
                refresh_message_box_bottom(text=text)

            refresh_text_box_loss(str(round(total_loss, 2)))

        refresh_message_box_top(text='')
        refresh_message_box_bottom(text='Training Finished')
    else:
        refresh_message_box_top(text='Network is not ready, please build network before training...')


def main():
    game_display.fill(BLACK)
    draw_controls()
    global CURRENT_NUM_LAYERS
    global NETWORK_ACTIVATIONS

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        check_controls()
        clock.tick(15)


if __name__ == '__main__':
    if False:
        import pygame._view

    main()
