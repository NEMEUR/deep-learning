
Ce repertoire  inculut en premier lieu différents rappels sur le deep learning et en second lieu des exemples de projets de deep learning sur lesquels je me suis amusée à reproduire.

# Rappels sur le deep-learning:
+ Un réseau artificial de neurones -> regression et classification
+ Un ANN est constitué de signaux d’entrées (variables), une ou plusieurs couches cachées de neurones et un signal de sortie(prédiction)
+ Feature scaling: les variables d'entrées doivent être sur la même échelle de valeurs cad standardiser et normaliser.
+ standarisation: x_stand=x-mean(x)/standard deviation(x)  [mean=0, equart type=1]
+ normalisation:x_norm=x-min(x)/max(x)-min(x)
+ Les variables d'entrées . 
+ la prediction peut être de 3 catégories :
  + continue (predire un prix).
  + binaire(oui ou non).
  + catégorique.
+ Pour chaque valeur des variables d'entrées X = chaque ligne d’observation --> on applique le réseau de neurone pour avoir la prédiction y de cette l’observation.
+ Ces signaux d'entrées (X) doivent être pondérés (w) et sommés pour avoir une sortie de signal (∑▒wXi)
+ Entrainer un réseau de neurone est en résumé ajuster les poids qui sont attribués aux signaux d’entrée qui permettront d'activer ou non les neurones du réseau (propagation).
+ Il ya plusieurs types de fonctions d’activation de neurones, les plus utilisés sont:
  +Fonction seuil (x<0 --> y=0, si x>=0 ->y=1)
  +Fonction sygmoid 1/(1+ e^(-x)) = fonction qui ressemble à la régression logistique (X la somme des variables) : c'est une probabilité [0-1]
  +Fonction Redressor=max(x,0) si x<0 on revoit 0 si x>=0 on revoit x
  +Tangente hyperboliqe (tanh) =(1-e^(-x))/(1+e^(-2x)) fonction qui ressemble à la fonction sygmoid mais la probabilité est entre [-1,1]

# Apprentissage des ANN :
+ Initialiser les poids avec les valeurs proches de 0.
+ On envoie des exemples d'observations (valeurs d'entrées X avec nos variables d’entrées, une variable par neurone).
+ Propagation avant : activation des neurones selon le poids attribués (choix de la fonction d’activation) et prédiction.
+ retro- propagation : calcul de l’erreur par rapport à la prédiction et un taux d’apprentissage  pour savoir ajuster et mettre à jour les poids.
+ Répétions des étapes d’apprentissages dans le but de minimiser la fonction de coût, on fait soit un apprentissage par lot (gradient descent) ou bien l’apprentissage par renforcement (l’algorithme de gradient stochastrique /iterative).
+ Quand on a entrainer notre ANN avec tout le jeu de données est passée : propagation + retro propagation, on appelle ça une époc.

# Les modules ANN:
+ from keras.models import Sequential #module qui initialise les ANN
+ from keras.layers import Dense # creation des couches des ANN

# Initialisation d'un ANN:
+ classifier1=Sequential() 

# Ajout de couches:
+ classifier1.add(Dense(units=??,activation="relu",kernel_initializer="uniform",input_dim=??)) # la couche d'entrée est paramétrée une seule fois input_dim= .., relu = fonction redresseur et unit contient le nombre de neurones dans la couche cachée 
+ classifier1.add(Dense(units=??,activation="relu",kernel_initializer="uniform")) #ajouter une deuxième couche cachée
+ classifier1.add(Dense(units=???,activation="sigmoid",kernel_initializer="uniform")) # ajoute une couche de sortie units= nb de variables à prédire,  exple de fonction de sortie:  sigmoide, si plusieurs variables de sorties activation= softmax  

# compilation d'un ANN
+ classifier1.compile(optimizer="adam", loss="binary_crossentropy",metrics=["accuracy"]) #optimiser="adam" -->l'algorithme gradient stochastique , loss=fonction de coût (logarithmique) "binary_crossentropy" et si on a plusieurs catégories de sorties on utilise loss="categorical_crossentropy", metrics mesure la performance du modèle.


# Amélioration d'un ANN :
+ Plus la fonction de coût diminue (cad l’erreur sur la prédiction est petite) plus la précision du modèle augmente.
+ la précision change à chaque fois qu'on entraine le modèle ML --> c’est dû au concept # biais-variance # --> on veut un modèle précis dans ces prédictions cad biais faible (erreur) et un modèle souvent précis (variance faible) 
+ pour résoudre le problème de variance élévée ,on applique la techinque K-fold cross validation cad qu' on prends le training set et on divise en k training folds ( avec une 1 portion test et k-1 portion train) à la fin on a k valeurs pour la précision et on fait le (mean de ces valeurs).
+ On étudie cette moyenne de précision par rapport à toutes les précisions (cad l’equart type) pour voir si ces précisions sont proches (variance faible) ou plutôt très dispersés ( haute variance)
+ Un autre paramètre d’amélioration d’un ANN : régularisation # dropout # c'est une technique pour résoudre le surapprentissage (overfiting) des ANN qui cause une variance très elevée dû que le modèle ML est très bon sur le train mais mauvais sur le test --> on a une précision sur le train est plus élevée que sur le test.
+ dropout est la technique qui permet de supprimer la dépendance qui peut se créer entre les neurones au moment de l'apprentissage.
+ Optimisation des hyperparamètres permet d'augmenter la précision des ANN --> on applique plusieurs combinaisons des hyper paramètres avec l’algortithme GridSearchCV.

# Réseau de neurones à convolution CNN : 
+ Réseau de neurones à convolution --> utiliser pour la classification des images, computer vision
+ Les étapes de fonctionnement des CNN sont :
  + Etape 1.a "convolution" : 
     + On a une entrée ex. image --> on applique la convolution (features detectors) --> but : detecter des features --> enregistrer dans des feautures maps --> sortie couche convolution (convolutional layer).

  + Etape 1.b "couche Relu" : 
     + on utilise la fonction redressor car on a besoin d'une fonction non linéaire pour nos features detectors (les images sont des objets non linéaires) --> on remplace toutes les valeurs négatives par des 0.
  + Etape2 "pooling" : 
     + objectif: éviter le sur-entrainement de notre CNN car  on supprime l’information qui n’est pas importante 
     + en entrée:  chaque feauture map --> on applique le max pooling (max des valeurs) --> pooled feature map --> sortie pooling layer.
  + Etape3 "flattening" : 
     + transformer le pooling layer (les pooled feature map) en couche d'entrée dans un ANN
  + Etape4 "ajouter un ANN complétement connecté"
     + ajouter un ANN où les couches cachées sont complétement connectées.
+ Pour améliorer notre CNN (précision du modèle):
  + On ajoute une couche de convolution
  + On ajoute une couche de complétement connecté
  + On ajoute la couche de convolution et celle complétement connecté.


# Réseaux de neuronnes récurrents: RNN
+ Un Réseau de neurones récurrents --> séries temporelles
+ Un RNN peut avoir deux problèmes:
  +  vanishing gradient :Wrec est petit <1 --> gradient petit qui empeche l’entrainement du réseau
  +  exploding gradient: Wrec est grand >1 
+ Solutions :
  + Exploding gradient :
    - retropropagation  tronquée
    - pénaliser le gradient : réguler
    - gradient clipping : on met un seuil pour le gradient qui ne peut pas dépasser (max)

  + Vanishing gradient :
    - initaliser les poids au départ
    - echo state network
    - LSTM (large mémoire court-time)


