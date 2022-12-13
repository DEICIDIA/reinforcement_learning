# Import des bibliothèques nécessaires
using PyCall
using Flux, Gym

# Création de l'environnement de simulation
env = GymEnv("CartPole-v0")

# Définition du modèle de réseau de neurones
model = Chain(
    Dense(env.observation_space.shape[1], 24, relu),
    Dense(24, 24, relu),
    Dense(24, env.action_space.n)
)

# Compilation du modèle avec l'algorithme d'optimisation Adam
opt = Adam(params(model), 0.001)

# Boucle d'entraînement
for episode in 1:100
    # Réinitialisation de l'environnement
    state = env.reset()

    # Boucle d'exécution d'une épisode
    while true
        # Afficher l'environnement (optionnel)
        env.render()

        # Prédire l'action à effectuer à partir de l'état actuel
        action = argmax(model(state))

        # Exécuter l'action dans l'environnement et obtenir la récompense et le prochain état
        (next_state, reward, done, _) = env.step(action)

        # Si l'épisode est terminé, arrêter la boucle
        if done
            break

        # Mettre à jour l'état actuel
        state = next_state
    end
end

# Fermer l'environnement de simulation
env.close()
