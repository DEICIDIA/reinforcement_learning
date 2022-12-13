# Définir le nombre d'états et d'actions
n_states = 10
n_actions = 4

# Initialiser la fonction de valeur Q à zéro
Q = zeros(n_states, n_actions)

# Définir le taux d'apprentissage (α) et la récompense de disconfort (γ)
α = 0.1
γ = 0.9

# Définir la fonction de mise à jour de la fonction de valeur Q
function update_Q(state, action, reward, next_state)
    Q[state, action] = Q[state, action] + α  * (reward + γ * maximum(Q[next_state, :]) - Q[state, action])
end

# Utiliser la fonction de mise à jour de Q pour entraîner l'algorithme
for i in 1:1000
    # Choisir un état et une action de manière aléatoire
    state = rand(1:n_states)
    action = rand(1:n_actions)
    
    # Simuler l'obtention d'une récompense et la transition vers l'état suivant
    reward = rand()
    next_state = rand(1:n_states)
    
    # Mettre à jour la fonction de valeur Q
    update_Q(state, action, reward, next_state)
end

# Utiliser la fonction de valeur Q pour prédire la meilleure action à prendre dans chaque état
for state in 1:n_states
    best_action = findmax(Q[state, :])
    println("Pour l'état $state, la meilleure action est $best_action")
end

