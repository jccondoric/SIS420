import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import math

# --- Funciones de Discretización (Iguales que antes) ---
def create_bins(num_bins=10):
    pi = math.pi
    return [
        np.linspace(-1.0, 1.0, num_bins),        # cos(theta1)
        np.linspace(-1.0, 1.0, num_bins),        # sin(theta1)
        np.linspace(-1.0, 1.0, num_bins),        # cos(theta2)
        np.linspace(-1.0, 1.0, num_bins),        # sin(theta2)
        np.linspace(-4 * pi, 4 * pi, num_bins),  # velocidad angular 1
        np.linspace(-9 * pi, 9 * pi, num_bins)   # velocidad angular 2
    ]

def discretize_state(observation, bins):
    discrete_state = []
    for i in range(len(observation)):
        idx = np.digitize(observation[i], bins[i]) - 1
        idx = max(0, min(idx, len(bins[i]) - 1))
        discrete_state.append(idx)
    return tuple(discrete_state)

# --- Función de Entrenamiento VISUAL ---
def train_and_show_acrobot(episodes):
    # 1. Creamos el entorno CON INTERFAZ VISUAL DESDE EL INICIO
    # Nota: Acrobot-v1 no acepta render_mode en el reset(), debe ser en el make()
    print("Iniciando simulación visual y entrenamiento...")
    env = gym.make('Acrobot-v1', render_mode='human')
    
    num_bins = 10
    bins = create_bins(num_bins)
    
    # Inicialización de la Tabla Q
    q_table_shape = [num_bins] * 6 + [env.action_space.n]
    q_table = np.zeros(q_table_shape)
              
    learning_rate = 0.1
    discount_factor = 0.99
    epsilon = 1.0
    epsilon_decay_rate = 1.0 / episodes # Decaimiento adaptado al nº de episodios
    rng = np.random.default_rng()
    
    # Array para recompensas
    rewards_per_episode = np.zeros(episodes)
    
    for i in range(episodes):
        # Reiniciar entorno y obtener estado discretizado
        observation, info = env.reset()
        state = discretize_state(observation, bins)
        
        terminated = False
        truncated = False
        total_reward = 0
        
        # Bucle de pasos (el renderizado ocurre automáticamente por el render_mode='human')
        while not terminated and not truncated:
            # Exploración vs Explotación
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
                
            new_observation, reward, terminated, truncated, _ = env.step(action)
            new_state = discretize_state(new_observation, bins)
            
            # --- CONTROL DE VELOCIDAD DE ANIMACIÓN (Opcional) ---
            # Si la animación va demasiado rápido, descomenta la siguiente línea:
            # time.sleep(0.01) 
            
            # Actualizar la Tabla Q
            best_future_q = np.max(q_table[new_state])
            q_table[state][action] += learning_rate * (reward + discount_factor * best_future_q - q_table[state][action])
            
            state = new_state
            total_reward += reward
            
        # Reducir epsilon
        epsilon = max(epsilon - epsilon_decay_rate, 0.01)
        
        # Registrar recompensa total
        rewards_per_episode[i] = total_reward
        
        # Imprimir progreso más seguido ya que vamos lento
        if (i + 1) % 10 == 0:
            print(f'Episodio {i + 1}/{episodes} | Recompensa: {total_reward} | Epsilon: {epsilon:.2f}')
            
    env.close()
    
    # --- Generar la gráfica final ---
    print("Simulación terminada. Generando gráfica...")
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.mean(rewards_per_episode[max(0, t - 100):(t + 1)])
    
    plt.plot(sum_rewards)
    plt.title('Rendimiento Visual de Acrobot (Q-Learning Discretizado)')
    plt.xlabel('Episodios')
    plt.ylabel('Recompensa Media (últimos 100)')
    
    # Guardamos y mostramos la gráfica
    plt.savefig('grafica_acrobot_visual.png')
    plt.show()

if __name__ == '__main__':
    # IMPORTANTE: Hemos reducido los episodios a 100 para que la demo visual 
    # no tarde demasiado. Si quieres que aprenda de verdad a balancearse, 
    # necesitarás subir este número (por ejemplo a 2000 o más) y esperar mucho más tiempo.
    train_and_show_acrobot(1000)