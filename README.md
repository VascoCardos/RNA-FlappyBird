# Flappy NEAT

Um clone simplificado do Flappy Bird desenvolvido em **Python** com **Pygame**, integrado com **NEAT** para treinar uma rede neural capaz de jogar automaticamente.

## 📦 Requisitos
- Python 3.8+
- [Pygame](https://www.pygame.org/)
- [NEAT-Python](https://neat-python.readthedocs.io/)
- NumPy

Instalação das dependências:
```bash
pip install pygame neat-python numpy
```

## 🚀 Como usar

### Jogar manualmente
```bash
python flappy_neat.py play
```

### Treinar a IA
```bash
python flappy_neat.py train
```
Ou especificar número de gerações:
```bash
python flappy_neat.py train 50
```

### Assistir a IA jogar
Após treinar, para assistir o melhor agente:
```bash
python flappy_neat.py watch
```

## ⚙️ Funcionamento
- O jogo é controlado por uma rede neural feed-forward treinada pelo algoritmo NEAT.
- Entradas da rede: posição vertical do pássaro, velocidade, distância até o próximo cano, topo do gap, base do gap.
- Saída: decisão de pular ou não.
- Recompensas:
  - +0.1 por frame vivo
  - +5 ao passar um cano
  - Penalização leve ao morrer

## 📂 Estrutura
- `flappy_neat.py` → Script principal com jogo, treino e execução.
- `neat-config.txt` → Configuração do NEAT (gerada automaticamente se não existir).
- `best.pickle` → Melhor genoma salvo após treino.

## 📝 Licença
Projeto livre para uso educacional.
