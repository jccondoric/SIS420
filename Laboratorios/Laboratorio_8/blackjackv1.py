import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def train_blackjack(episodes=10000):
    env = gym.make('Blackjack-v1')
    
    # Blackjack: 360 estados posibles (suma 4-21, carta 1-10, as usable sí/no)
    q_table = np.zeros((360, 2))  # 2 acciones: 0=pedir, 1=plantarse
    
    learning_rate = 0.1
    discount_factor = 0.95
    epsilon = 1
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()
    
    rewards_per_episode = np.zeros(episodes)
    
    # Convierte la tupla del estado a un número del 0 al 359
    def convertir_estado(estado):
        suma, carta, as_usable = estado
        idx_suma = max(0, min(suma - 4, 17))     # suma: 4-21 -> 0-17
        idx_carta = max(0, min(carta - 1, 9))    # carta: 1-10 -> 0-9
        idx_as = 1 if as_usable else 0
        return idx_suma + idx_carta * 18 + idx_as * 180
    
    for i in range(episodes):
        if (i + 1) % 500 == 0:
            env.close()
            env = gym.make('Blackjack-v1', render_mode='human')
        else:
            env.close()
            env = gym.make('Blackjack-v1')
        
        state = convertir_estado(env.reset()[0])
        terminated = False
        truncated = False
        
        while (not terminated and not truncated):
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])
                
            next_state_raw, reward, terminated, truncated, _ = env.step(action)
            next_state = convertir_estado(next_state_raw)
            
            # Fórmula Q-Learning IGUAL a la tuya
            q_table[state, action] = q_table[state, action] + learning_rate * (
                reward + discount_factor * np.max(q_table[next_state, :]) - q_table[state, action]
            )
            
            state = next_state
        
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        
        if reward == 1:
            rewards_per_episode[i] = 1
        
        if (i + 1) % 500 == 0:
            print(f'Episodio {i + 1} | Win rate: {np.sum(rewards_per_episode[i-499:i+1])/5:.1f}%')
    
    env.close()
    
    print(f'Tasa de victorias total: {np.sum(rewards_per_episode)/episodes*100:.1f}%')
    
    # Gráfica IGUAL a la tuya
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):(t + 1)])
    
    plt.plot(sum_rewards)
    plt.show()

if __name__ == '__main__':
    train_blackjack(10000)