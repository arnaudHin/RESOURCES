#AUTOR : Arnaud HINCELIN
#Date : 07/2020


print("==================SCIPY======================")
"""
Outils mathematiques très puissants qui permet l'interpolation,traitement du signal,
(Fourier), 
"""

# 1 - PARTIE INTERPOLATE
"""
situation : datasets où il manque des valeus, par exemple car entre deux
capteurs il n'y a pas la même fréquence d'acquisition. 
Pour rajouter des valeurs dans notre dataset, il faut interpoler.

importer la fonction interpld du package scipy.interpolate

la fonction interpolate va prmettre de générer une autre fonction
(fction d'interpolation).
Plusieurs types d'interpolation : 
Linéaire : ligne entre deux points 
Cubique : courbe entre deux points (adaptée pour oscillations,...)
pleins d'autres... 

En para : données à interpoler : x et y ; 
		  type d'interpolation "kind" : linear, cubic


ATTENTION, en interpolation on ne recréé pas la réalité !! 
il peut très bien y avoir des oscillations invisibles, non vues
à cause d'une fréquence d'acquisition trop faible

"""
import numpy as np
import matplotlib.pyplot as plt #import pyplot
from scipy.interpolate import interp1d

x = np.linspace(0, 10, 30) #30 nbs de 0 à 10
y = x**2

new_x = np.linspace(0,10,30) #pareil que x

f = interp1d(x, y, kind='linear') #fonction interpolation inéaire
result = f(new_x) #new_x en fonction interpolé
f2 = interp1d(x, y, kind='cubic') #fonction interpolation inéaire
result2 = f2(new_x)

plt.figure()
plt.scatter(x, y) #y en focntion de x
plt.subplot(3,1,1)
plt.scatter(new_x, result, c='r') #interpolée en fonction de new_x
plt.subplot(3,1,2)
plt.scatter(new_x, result2)
plt.subplot(3,1,3)

plt.close()


#2 PARTIE OPTIMIZE
"""
On veut minimiser un dataset
Deux fonctions principales : 
minimize() et optimize()
Et on peut faire de la programmation linéaire, pr résoudre un pb
d'optimisation tout en rspectant certaines contraintes.

"""
#MODULE optimize()
"""
fonction curve_fit() : 
pour optimier le placement d'une courbe à travers un nuage de point

Ds exemple on a une fonction cubique avec du bruit (nuage de points désordonnés)
on veut générer un modèle statistique qui rentrer parfaitement
avec ce nuage de point.
fonction curve_fit() utilise la méthode des "moindes carrés" pr
trouver les meilleurs paramètres (a,b,c,...) d'un modèle fourni
à notre fonction.
Donc, comme notre nuage de point est caractérisé par une fonction
cubique, notre modèle sera de type ax**3 + bx**2 + cx +d

curve_fit(f, x, y)
f : modèle (fonction, ici cubique avec les param a,b,... )
x et y : données

Renvoi 2 tableaux numpy:
premier : les diff param du modèle (a,b,c,d)
deuxième : matrice de covariance du modèle

"""
from scipy import optimize
#dataset
x = np.linspace(0, 2, 200) #200 points de 0 à 2
#fonction cubique auquel on ajoute du bruit
y = 1/3*x**3 - 3/5*x**2 + 2 + np.random.randn(x.shape[0])/20

#définir un modèle cubique car nuage de point généré par cubique
def f(x,a,b,c,d):
	return a*x**3+b*x**2+c*x+d
#donner modèle et données à la fonction curve_fit
params, covar = optimize.curve_fit(f, x,y) #deux tableaux numpy


plt.figure()
plt.scatter(x, y) #nuage de points 
#fonction où y est notre fonction f avec les params du tableau numpy
plt.plot(x, f(x, params[0], params[1], params[2], params[3]), c='g', lw=3)

plt.close()


#MODULE Minimize()

"""
Cette fonction, va executer un algo de minimisation qui va, selon
un point de départ faire converger la fonction vers un minimum 
local depuis le point de départ.

Paramètres : 
fonction de départ
x0 : un point de départ

renvoi des infos, mais ce qui nous intéresse est surtout la var x
 qui corresspond au premier minimum local du point de départ

Et pour avoir minimum global : on change notre point de départ  

Mais minimize() fonctionne aussi avec fonctions à n dimensions
et cela devient très puissant ! 
Avec contourplot, minimum (violet) et maximum (jaune), on peut
utiliser minimize() pour visualiser les minimums !


"""
#1D
#création d'une fonction avec abscisses de -10 à 10
def f(x):
	return x**2 + 15*np.sin(x)
x = np.linspace(-10, 10, 100)

x0 = -4
optimize.minimize(f, x0 = -8).x #renvoi -6,... minimum local de -8
mimi_glo = optimize.minimize(f, x0 = -4).x #renvoi -1,... minimum local de -4
#mais c'est aussi le mimimum global ! 

#afficher 
plt.figure()
plt.plot(x, f(x), lw= 3, zorder = -1)#fonction
plt.scatter(mimi_glo, f(mimi_glo), s=100, c='r')#point minimum global
plt.scatter(x0 , f(x0), c='g', marker = '+',s=200) #point x0

plt.close()


#2D
#exemple avec un contour plot
def f(x):
	return np.sin(x[0]) + np.cos(x[0]+x[1])*np.cos(x[0])

x = np.linspace(-3,3, 100)
y = np.linspace(-3,3, 100)
x,y = np.meshgrid(x,y)

#on choisit le point départ au centre du contour
x0 = np.zeros((2,1))
#calcule du minimum local
result = optimize.minimize(f, x0=x0).x


#visualise
plt.figure()
plt.contour(x, y, f(np.array([x,y])), 20)
plt.scatter(x0[0], x0[1], marker = '+', c='r', s=100) #point depart
plt.scatter(result[0], result[1], c='g', s=100) #min local

plt.close()


#3 PARTIE TRAITEMENT SIGNAL
"""
Deux modules importants : 
scipy.signal : Signal processing (convolution, filtres,...) 
scipy.fftpack : Discrete Fourier transforms (transfo de fourier)


"""

#MODULE SIGNAL PROCESSING
"""
La fonction detrend() permet d'éliminer toute tendance linéaire 
d'un signal ; ex pour une fonction qui croit vers l'infini.

ordonnées_2 = signal.detrend(ordonnées)

"""
from scipy import signal
#dataset d'une fonction à tendance linéaire, petites oscillations 
#suivant une droite croissante
x = np.linspace(0,20,100)
y = x + 4*np.sin(x)+np.random.randn(x.shape[0])

#créer nouvelle coordonnées y
new_y = signal.detrend(y)
#afficher
plt.figure()
plt.plot(x,y) #fonction avec tendance linéaire
plt.plot(x, new_y) #fonction sans tendance linéaire

plt.close()

#MODULE DISCRETE FOURIER TRANSFO
"""
Transfo de Fourier : tech pour extraire et analyser les fréquences
présentes dans un signal périodique.

Par exemple, on a 3 signaux sinusoidales  périodiques.
En combinant ces 3 signaux, on obtient un signal réel que l'on
peut trouver partout (son, elec,...); car ondes = combinaison de fréquence

Puis on extrait les différentes fréquences qui le compose. 
On obtient donc un spectre, avec en abscisses les fréquences, 
et en ordonné les amplitude pour chaque fréquence (1 pic = 1 fréquence)
Donc si 3 signaux de départ => 3 pics

fonctions:
fft(y) : donne tableau fourier
fftfreq(taille de y) : donne tableau de fréquences 

"""
from scipy import fftpack

#génère un signal réel avec 3 fréquences diff
x = np.linspace(0,30,1000)
y = 3*np.sin(x)+2*np.sin(5*x)+np.sin(10*x)

#recupère fréquences et amplitude
fourier = fftpack.fft(y)
freq = fftpack.fftfreq(y.size)
#valeur absolue pr éviter négatifs
power = np.abs(fourier)
freq_abs = np.abs(freq)

plt.figure()
plt.plot(x, y)

plt.close()

plt.figure()
plt.plot(freq_abs, power) #spectre des 3 pics de fréquence du signal (x,y)

plt.close()

"""
Application infinies ! exemple : filtrer un signal
Un signal initial est perdu dans su bruit, pour le filtrer et enlever
le bruit en 3 étapes
-> Transfo de fourier
Pour produire le spectre du signal
Avec fonctions : fft() et fftfreq()

-> Boolean indexing
Pour enlever valeurs en dessous d'un certain seuil
Ainsi spectre net et lisse

-> Applique transfo de fourier inverse 
Pour repasser le signal en réel
Avec fonction : ifft()

"""
#génère signal avec du bruit
y = 3*np.sin(x)+2*np.sin(5*x)+np.sin(10*x) + np.random.randn(x.shape[0])

#plt.plot(x, y) #signal de départ avec bruit

#1 : transfo de fourier
fourier = np.abs(fftpack.fft(y))
freq = np.abs(fftpack.fftfreq(y.size))
plt.figure()
plt.plot(freq, fourier) #spectre de fourier bruité

#2 : boolean indexing
#en analysant spectre, on défini le seuil à 400
fourier[fourier<400] = 0
plt.plot(freq, fourier) #spectre de fourier net

plt.close()


#3 : Transfo de fourier inverse
filter_signal = fftpack.ifft(fourier)
plt.figure(figsize = (12,8))
plt.plot(x, y, lw=0.5) #signal original bruité
plt.plot(x, filter_signal, lw=3) #signal filtré

plt.close()

#plt.show()




#4 PARTIE TRAITEMENT IMAGE
"""
Le module ndimage permtet de manipuler complètement les images : 
filtrage, convolutions, interpolations, mesures, morphology

"""

# La MORPHOLOGY 
"""
Technique mathématique qui permet de transformer des matrices, 
et donc des images. 
On définir une structure (souvent une croix de pixel blancs), 
qui va se déplacer de pixel en pixel sur tte l'image de pixel noirs.
A chaque rencontre avec autre pixel blanc, elle va effectuer une opération.

Deux opérations :
Dilation : imprimer un pixel blanc
Erosion : efface des pixel blancs en pixel noir

Ainsi la structure va se déplacer et modifier les pixels par 
une opération selon ce qu'on lui demande.
Utile pour enlever des pixels parasites d'une image, mais modifie
un peu l'info de l'image


"""
#Exemple : enlever des pixels parasites d'une image
from scipy import ndimage
np.random.seed(0)
X = np.zeros((32,32)) #créé image de pixels violets
X[10:-10, 10:-10] = 1 #créé carré de pixels jaune au centre
#créé pixels jaune parsites autour du carré
X[np.random.randint(0, 32, 30), np.random.randint(0,32,30)]=1

#créé nlle image "déparasitée" grace à fonction binary_opening()
#cette fonction combine dilation et erosion

new_im = ndimage.binary_opening(X)

plt.figure()
plt.imshow(X)

plt.close()

plt.figure()
plt.imshow(new_im)

plt.close()


#SEGMENTER UNE IMAGE
"""
1 - fonction label() de scipy.ndimage permet de segmenter une image
et mettre une étiquette sur des objets de même couleur.

prend en param l'image et renvoi deux variables : 
- image avec les étiquettes sur chaque objet
- nombre d'étiquette

En fait, imahge renvoyée est une image où chaque objet est trié
selon sa taille et une couleur proportio à sa taille lui est 
attribué.
"""
label_image, nb_eti = ndimage.label(new_im)
nb_eti #affiche 1 car une seule étiquette (carré) sur cette image

"""
2 - fonction sum() qui va permettre de compter le nb de pixel de
chaque étiquette.

prend en param l'imge initiale, l'image des etiquettes, et le nb
d'étiquettes à éxaminer.
renvoi un tableau de taille du nombre d'étiquettes examinées,
avec la taille de chaque étiquette

"""
taille_eti = ndimage.sum(new_im, label_image, range(nb_eti))
#tad d'un unique element
#puis on peut afficher sur un graph
#plt.scatter(range(nb_eti), taille_eti, c='r')

#EXEMPLE REEL
"""
On charge une image médicale de bactérie 

1 --> Extraire les bactéries de l'arrière plan de la photo
2--> Utiliser morphologie pour enlever les parasites
3--> Mesurer taille de chaque bactérie et présenter résultats dans un graph

importer photo : plt.imread('nom.png')
on vérifie sa dimension, et comme dim = 3, on modifie en ne conservant 
que la 2D car plus simple à manipuler. Avec du subsetting


"""
#Importer image et vérifier la dimension du tab 
I = plt.imread('bacteria_cell.jpg')
I.shape #affiche (546, 728, 3) donc 3 dimensions
I = I[:,:,0]# passer en 2D, supprime la 3e dim
I.shape #affiche (546, 728, 2) donc 2 dimensions
#plt.imshow(I, cmap='gray') #affiche en 2D et en N&B

#1 --> Extraire bactéries
#Utilisons le boolean indexing mais avant créer copie d'image
I2 = np.copy(I)
#applatir copie pr en faire un histogramme
#plt.hist(I2.ravel(), bins=255) #affiche histogramme de l'image (majeure en 255 = blanc)
#Sur histo on observe 3 pics (100, 200 à 230 et un grabd à 255), normal car bcp de blanc
#à partir de l'histo, on fait un boolean indexing

I = I[0:520, :]#on coupe les dernières lignes car texte à enlever
I = I<130 #less pixels>130 sont mis en false dans notre tableau d'image

plt.figure()
plt.subplot(2,2,1)
plt.imshow(I)
#plt.imshow(I, cmap='gray')


#2 --> Morphology pour enlever les parasites
I3 = ndimage.binary_opening(I)

plt.subplot(2,2,2)
plt.imshow(I3)

#3 --> Taille des bacteries
label_image, nb_eti = ndimage.label(I3)
#Donc chaque bacté est bien représentée par une couleur, selon sa taille
#bleu == grand et jaune == petit
plt.subplot(2,2,3)
plt.imshow(label_image)

#nobre de bactérie
nb_eti#affiche 72, donc environ 72 bactéries

#calcul des tailles
taille_bac = ndimage.sum(I3, label_image, range(nb_eti))
#tab de 72 données

#affiche tailles dans un grap
plt.figure()
plt.scatter(range(nb_eti), taille_bac, c='r')

plt.show()






