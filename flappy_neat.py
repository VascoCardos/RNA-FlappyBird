"""
Flappy Bird minimal em pygame + treino NEAT
Arquivos: este script único criará um ficheiro de configuração NEAT se não existir.

Requisitos:
  pip install pygame neat-python numpy

Uso:
  python flappy_neat.py play    # jogar manualmente
  python flappy_neat.py train   # treinar com NEAT (vai criar neat-config.txt se não existir)
  python flappy_neat.py watch   # ver o melhor genome guardado (best.pickle)

O script inclui um modo simples de estado para a rede neural:
  inputs: [bird.y, bird.vel, distance_to_pipe, top_pipe_y, bottom_pipe_y]
  output: pular se > 0.5

Nota: Este código é intencionalmente simples/educativo — ótimo para começar!
"""

import os
import sys
import random
import math
import pickle
import time

import pygame
import neat
import numpy as np

# --------------------------- Configurações ---------------------------
WIN_WIDTH = 400
WIN_HEIGHT = 600
FPS = 30

# física do pássaro
GRAVITY = 1
JUMP_VEL = -10
MAX_DROP_SPEED = 10

# canos
PIPE_GAP = 150
PIPE_WIDTH = 80
PIPE_VELOCITY = 4
PIPE_FREQ = 90  # frames entre canos

# arquivos
NEAT_CONFIG_FILE = 'neat-config.txt'
BEST_GENOME_FILE = 'best.pickle'

# --------------------------------------------------------------------
# Recursos visuais - superfícies simples (retângulos) para manter curto
# --------------------------------------------------------------------

def draw_text(surface, text, size, x, y):
    font = pygame.font.SysFont('Arial', size)
    text_surf = font.render(text, True, (255, 255, 255))
    surface.blit(text_surf, (x, y))


class Bird:
    WIDTH = 34
    HEIGHT = 24

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vel = 0
        self.alive = True
        self.score = 0

    def jump(self):
        self.vel = JUMP_VEL

    def update(self):
        self.vel += GRAVITY
        if self.vel > MAX_DROP_SPEED:
            self.vel = MAX_DROP_SPEED
        self.y += self.vel

    def get_mask(self):
        # mask used only if you add pixel-perfect collisions
        return None

    def draw(self, win):
        pygame.draw.rect(win, (255, 255, 0), (int(self.x), int(self.y), self.WIDTH, self.HEIGHT))


class Pipe:
    def __init__(self, x):
        self.x = x
        self.width = PIPE_WIDTH
        self.gap = PIPE_GAP
        self.top = 0
        self.bottom = 0
        self.passed = False
        self.set_height()

    def set_height(self):
        # top pipe bottom y
        center = random.randint(int(WIN_HEIGHT * 0.3), int(WIN_HEIGHT * 0.7))
        self.top = center - self.gap // 2 - WIN_HEIGHT  # top rect y
        self.bottom = center + self.gap // 2  # bottom rect y

    def update(self):
        self.x -= PIPE_VELOCITY

    def collide(self, bird: Bird):
        bx, by = bird.x, bird.y
        bw, bh = bird.WIDTH, bird.HEIGHT
        # top pipe rect: (x, top, width, WIN_HEIGHT) but top is negative
        top_rect = pygame.Rect(self.x, self.top, self.width, WIN_HEIGHT)
        bottom_rect = pygame.Rect(self.x, self.bottom, self.width, WIN_HEIGHT)
        bird_rect = pygame.Rect(bx, by, bw, bh)
        return bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect)

    def draw(self, win):
        # top pipe (drawn from top of screen downwards)
        pygame.draw.rect(win, (0, 255, 0), (int(self.x), 0, self.width, int(self.top + WIN_HEIGHT)))
        # bottom pipe
        pygame.draw.rect(win, (0, 255, 0), (int(self.x), int(self.bottom), self.width, WIN_HEIGHT - int(self.bottom)))


class Base:
    HEIGHT = 100

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = WIN_WIDTH

    def update(self):
        self.x1 -= PIPE_VELOCITY
        self.x2 -= PIPE_VELOCITY
        if self.x1 + WIN_WIDTH < 0:
            self.x1 = self.x2 + WIN_WIDTH
        if self.x2 + WIN_WIDTH < 0:
            self.x2 = self.x1 + WIN_WIDTH

    def draw(self, win):
        pygame.draw.rect(win, (139, 69, 19), (self.x1, self.y, WIN_WIDTH, self.HEIGHT))
        pygame.draw.rect(win, (139, 69, 19), (self.x2, self.y, WIN_WIDTH, self.HEIGHT))


# --------------------------- Jogo principal ---------------------------

class FlappyGame:
    def __init__(self, render=True):
        pygame.init()
        self.render = render
        if render:
            self.win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
            pygame.display.set_caption('Flappy NEAT')
        else:
            # Headless pygame (still needs display) - but for simplicity we'll create a surface
            self.win = pygame.Surface((WIN_WIDTH, WIN_HEIGHT))

        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.bird = Bird(50, WIN_HEIGHT // 2)
        self.base = Base(WIN_HEIGHT - Base.HEIGHT)
        self.pipes = []
        self.frame_count = 0
        self.score = 0

    def step(self, action_jump=False):
        # action_jump: boolean
        # retorna (alive:boolean, reward:float, done:boolean)
        if action_jump:
            self.bird.jump()

        # atualizações
        self.bird.update()
        remove = []
        add_pipe = False

        for pipe in self.pipes:
            pipe.update()
            # check for passing
            if not pipe.passed and pipe.x + pipe.width < self.bird.x:
                pipe.passed = True
                self.score += 1
                self.bird.score += 1
            if pipe.x + pipe.width < 0:
                remove.append(pipe)

        for r in remove:
            try:
                self.pipes.remove(r)
            except ValueError:
                pass

        # spawn pipes
        if self.frame_count % PIPE_FREQ == 0:
            self.pipes.append(Pipe(WIN_WIDTH + 10))

        self.base.update()
        self.frame_count += 1

        # collisions
        if self.bird.y + self.bird.HEIGHT >= self.base.y:
            self.bird.alive = False
        if self.bird.y <= 0:
            self.bird.y = 0
            self.bird.vel = 0

        for pipe in self.pipes:
            if pipe.collide(self.bird):
                self.bird.alive = False

        done = not self.bird.alive
        reward = 0.0
        if done:
            reward = -1.0
        else:
            # small positive reward for staying alive, bigger for passing
            reward = 0.1 + (1.0 if self.score > 0 and self.pipes and self.pipes[0].passed else 0.0)

        return self.bird.alive, reward, done

    def get_state(self):
        # Retorna vector de entradas para a RN
        # Encontra o próximo cano (o primeiro com x + width > bird.x)
        next_pipe = None
        for pipe in self.pipes:
            if pipe.x + pipe.width > self.bird.x:
                next_pipe = pipe
                break
        if next_pipe is None:
            # valores padrão se não houver cano à frente
            dist = WIN_WIDTH
            top_y = 0
            bottom_y = WIN_HEIGHT
        else:
            dist = next_pipe.x - self.bird.x
            top_y = next_pipe.top + WIN_HEIGHT  # convertendo para y do topo do gap
            bottom_y = next_pipe.bottom

        # normalizar entradas
        # [bird_y_norm, bird_vel_norm, dist_norm, top_y_norm, bottom_y_norm]
        inputs = np.array([
            self.bird.y / WIN_HEIGHT,
            (self.bird.vel + MAX_DROP_SPEED) / (2 * MAX_DROP_SPEED),
            dist / WIN_WIDTH,
            top_y / WIN_HEIGHT,
            bottom_y / WIN_HEIGHT
        ], dtype=float)
        return inputs

    def render_frame(self):
        if not self.render:
            return
        self.win.fill((135, 206, 235))  # sky
        for pipe in self.pipes:
            pipe.draw(self.win)
        self.base.draw(self.win)
        self.bird.draw(self.win)
        draw_text(self.win, f'Score: {self.score}', 24, 10, 10)
        pygame.display.update()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()


# --------------------------- NEAT integration ---------------------------

# cria um ficheiro de configuração NEAT básico se não existir
DEFAULT_NEAT_CONFIG = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 10000.0
pop_size              = 50
reset_on_extinction   = False

[DefaultGenome]
# node activation options
activation_default      = sigmoid
activation_mutate_rate  = 0.0
activation_options      = sigmoid

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

conn_add_prob           = 0.5
conn_delete_prob        = 0.3

enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
initial_connection      = full_direct

node_add_prob           = 0.2
node_delete_prob        = 0.1

num_hidden              = 0
num_inputs              = 5
num_outputs             = 1

response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 15
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""


def ensure_neat_config(path=NEAT_CONFIG_FILE):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write(DEFAULT_NEAT_CONFIG)
        print(f'Criado ficheiro de configuração NEAT: {path}')


def eval_genomes(genomes, config):
    # jogar uma geração inteira — cada genoma corresponde a um jogador
    nets = []
    birds = []
    ge = []

    # criar instância do jogo para a população (headless render False)
    game = FlappyGame(render=False)

    for gid, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(50, WIN_HEIGHT // 2))
        ge.append(genome)

    pipes = []
    base = Base(WIN_HEIGHT - Base.HEIGHT)
    frame_count = 0
    score = 0
    alive_count = len(birds)

    # Give every genome up to a maximum number of frames or until all die
    max_frames = 2000
    while len(birds) > 0 and frame_count < max_frames:
        frame_count += 1
        # spawn pipes periodically
        if frame_count % PIPE_FREQ == 0:
            pipes.append(Pipe(WIN_WIDTH + 10))

        # Loop backwards for safe removal
        for i, bird in list(enumerate(birds))[::-1]:
            # create a temporary FlappyGame state to compute inputs
            temp_game = FlappyGame(render=False)
            temp_game.bird = bird
            temp_game.pipes = pipes

            inputs = temp_game.get_state()
            output = nets[i].activate(inputs)
            action_jump = output[0] > 0.5

            # apply physics
            if action_jump:
                bird.jump()
            bird.update()

            # check for collisions with base
            if bird.y + bird.HEIGHT >= base.y:
                ge[i].fitness += -1.0
                # remove bird
                del birds[i]
                del nets[i]
                del ge[i]
                continue

            if bird.y <= 0:
                bird.y = 0
                bird.vel = 0

        # update pipes and check collisions
        for pipe in list(pipes):
            pipe.update()
            if pipe.x + pipe.width < 0:
                try:
                    pipes.remove(pipe)
                except ValueError:
                    pass

            for i, bird in list(enumerate(birds))[::-1]:
                if pipe.collide(bird):
                    del birds[i]
                    del nets[i]
                    del ge[i]
                    continue

            # award point for passing
            for bird in birds:
                if not pipe.passed and pipe.x + pipe.width < bird.x:
                    pipe.passed = True
                    for g in ge:
                        g.fitness += 10.0

        base.update()

        # survive reward
        for g in ge:
            g.fitness += 0.1

    # end loop
    # ensure leftover genomes get their fitness translated
    # (nothing special needed)


def run_neat(config_file=NEAT_CONFIG_FILE, generations=30):
    ensure_neat_config(config_file)
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, generations)
    # salvar o melhor
    with open(BEST_GENOME_FILE, 'wb') as f:
        pickle.dump(winner, f)
    print('Treino terminado. Melhor genoma salvo em', BEST_GENOME_FILE)


def watch_best(genome_path=BEST_GENOME_FILE, config_file=NEAT_CONFIG_FILE):
    if not os.path.exists(genome_path):
        print('Nenhum genoma encontrado. Treine primeiro com `python flappy_neat.py train`')
        return
    ensure_neat_config(config_file)
    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    game = FlappyGame(render=True)
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        # estado atual -> decisão
        inputs = game.get_state()
        output = net.activate(inputs)
        action_jump = output[0] > 0.5

        # passo do jogo controlado pela IA
        alive, reward, done = game.step(action_jump)

        # renderização normal
        game.render_frame()

        # se morrer, reinicia para ver de novo
        if done:
            time.sleep(1)
            game.reset()

    game.close()



# --------------------------- Modo jogador manual ---------------------------

def play_manual():
    game = FlappyGame(render=True)
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    game.bird.jump()
        alive, reward, done = game.step(False)
        game.render_frame()
        if done:
            time.sleep(1)
            game.reset()
    game.close()


# --------------------------- Entrypoint ---------------------------

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Uso: python flappy_neat.py [play|train|watch]')
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == 'play':
        play_manual()
    elif cmd == 'train':
        generations = 30
        if len(sys.argv) >= 3:
            try:
                generations = int(sys.argv[2])
            except:
                pass
        run_neat(NEAT_CONFIG_FILE, generations=generations)
    elif cmd == 'watch':
        watch_best(BEST_GENOME_FILE, NEAT_CONFIG_FILE)
    else:
        print('Comando desconhecido.')
