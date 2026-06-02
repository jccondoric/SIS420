import gymnasium as gym
import time

# 1. Crear el entorno con la interfaz visual activa
env = gym.make("Acrobot-v1", render_mode="human")

# 2. Reiniciar usando tu semilla (123) y modificando los límites de inclinación inicial
# Esto hace que el péndulo comience con una oscilación inicial más fuerte
observation, info = env.reset(seed=123, options={"low": -0.2, "high": 0.2})

print("Opening Acrobot simulation window with seed 123...")

# 3. Ejecutar la simulación por 300 pasos
for step in range(300):
    # Seleccionar una acción aleatoria (-1, 0, o 1 de fuerza en el codo)
    action = env.action_space.sample()
    
    # Aplicar la acción seleccionada
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Pausa corta para que el ojo humano pueda seguir la animación de forma fluida
    time.sleep(0.02)
    
    # Si alcanza el objetivo o se agota el tiempo, reiniciar con la misma configuración
    if terminated or truncated:
        observation, info = env.reset(seed=123, options={"low": -0.2, "high": 0.2})

# 4. Cerrar la ventana de forma segura
env.close()
print("Simulation finished successfully!")
