# deep-learning

Ce repertoire  inculut les différents projets de deep learning sur lesquels je me suis entrainée.

#Rappel de deep-learning:

+ Un réseau artificial de neurones ANN est constitué de signaux (variables) d’entrées , une ou plusieurs couches cachées de neurones et un signal de sortie(prédiction)
+ Les variables d'entrées doivent être sur la même échelle de valeurs exple [-2,2] cad standardiser et normaliser sur la même echelle. 
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

Apprentissage des ANN :
+ Initialiser les poids avec les valeurs proches de 0.
+ On envoie des exemples d'observations (valeurs d'entrées X avec nos variables d’entrées, une variable par neurone).
+ Propagation avant : activation des neurones selon le poids attribués (choix de la fonction d’activation) et prédiction.
+ retro- propagation : calcul de l’erreur par rapport à la prédiction et un taux d’apprentissage  pour savoir ajuster et mettre à jour les poids.
+ Répétions des étapes d’apprentissages dans le but de minimiser la fonction de coût, on fait soit un apprentissage par lot (gradient descent) ou bien l’apprentissage par renforcement (l’algorithme de gradient stochastrique /iterative).
+ Quand on a entrainer notre ANN avec tout le jeu de données est passée : propagation + retro propagation, on appelle ça une époc.

