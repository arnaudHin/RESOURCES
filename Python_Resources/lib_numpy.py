#AUTOR : Arnaud HINCELIN
#Date : 07/2020

print("==========NUMPY AND MACHINE LEARNING===========")
"""
On commence ! 
package le plus important, numpy avec l'objet le plus puissant : 
tableau à n dimensiosn = ND Array
C'est dans ce tableau que l'on va entrer nos données pour entrainer notre IA
"""

#OBJET ND ARRAY
"""
Créer tableaux à 1(ligne) , 2(matrice) ou 3 dimensions (cube)

Tableau 1D ARRAY
ressemble à une liste (aussi des index)
mais ndarray bcp plus puissant : rapide, moins mémoire, plus de méthodes

Tableau 2D ARRAY
très utilisé, comparable à tableau excel
On peut stocker pixels d'une image noir et blanc, valeur de 0 à 255

Tableau 3D ARRAY
Alors on peut traiter images en couleur car superposition de 3 tableaux 2D
les 3 tableux 2D : vert, bleu ,rouge


Chaque tableau aura des méthodes et attributs :
attribut les plus utiliés :

#Attribut ndim
renvoi la dimension

#Attribut shape
renseigne sur la forme, combien lignes et colones.
shape renvoi un tuple:
séquence=on peut accéder aux diff valeurs (shape[n])
tuple=non modifiable (immutable)
tab à 1 dimension de 2 lignes
alors : shape=(2)
shape[0]=2
tab à 2 dimension de 3 colones et 2 lignes
alors : shape=(2,3)
shape[1]=3
tab à 3 dimension de 3 colones et 2 lignes et 2 tableaux superposés
alors : shape=(2, 3, 2)
shape[2]=2

"""

print("=====CONSTRUCTOR AND TABLEAUX======")


#Constructeurs de tableaux
"""
np.array(objet)
np.zeros(shape) #créer tableau init à 0
np.full(shape, value) #créer tableau init à value
np.ones(shape) #créer tableau init à 1
np.eye(n) #créé mat diagonale, avec n diagonales

np.random.randn(shape) #créer tableau de valeurs aléatoires
ces nbs aléa sont données selon la loi normale centrée en 0;proba plus importante pr nbs proches de 0

np.linspace(début,fin,quantité) #créé tab 1D dont on précise val de début
et val de fin, et la quantité déléments t.riés par ordre

np.arange(début, fin, pas) #créé tab 1D dont on précise val de début et 
val de fin, et le pas. On connait pas la qté d'éléments

Attention; shape est un tuple ! s'écrit np.zeros((2,3))
Sauf random : np.random.randn(2,3)



"""
import numpy as np
A=np.array([1,2,3])

A.shape #renvoi la forme
A.ndim #renvoi la dimension
A.size #renvoi nb éléments

B=np.zeros((3,2,3))#tableau de dim 3 init à 0
C=np.random.randn(2,3) #tableau de dim 2 init aléatoire
D=np.eye(3)
E=np.linspace(0,10,20,dtype=np.float16)


#Types de données
"""
On peut choisir le type de données des tableaux
np.float, np.int,... avec la param (dtype=np.float)
Mais aussi la place mémoire de chaque varibale
np.int8 ; np.int16 ; ... ; np.uint8 ; ...
mais,
np.int8 = moins précis, plus rapide
np.int64 = plus précis, mois rapide

"""
E=np.linspace(0,10,20,dtype=np.float16)


#Methodes importantes
A=np.zeros((3,2))
B=np.ones((3,2))

"""
#Coller 2 tableaux
np.hstack((Array A, Array B))#coller 2 tab horizontale, A et B même nb lignes
np.vstack((Array A, Array B))#coller 2 tab verticale, A et B même nb colones
np.concatenate((Array A, Array B), axis)#axis(axe de collage) = 0(axe verti) ou 1(axe hori) 

#Remanipuler forme d'un tableau
Il faut même nb éléments !
Array.np.reshape((a,b))
Pb des listes dim 1D : shape (3,)=> rien après virgule
pr éviter pb en algo, mettre un 1 pour remplacer espace après virgule
Faire reshape : Array.reshape((Array.shape[0], 1))

#Applatir un tableau en un tableau en 1D
Array.ravel()
[[0. 0.]		
 [0. 1.]	==>> [0. 0. 0. 1. 1. 1.]  
 [1. 1.]]


Si on veut ensuite utiliser (3,) => Array.squeeze()

"""
C=np.hstack((A,B)) #A et B même shape[0] !
C=np.vstack((A,B)) #A et B même shape[1] !
C=np.concatenate((A,B), axis=0) #comme hstack
C=C.reshape((3,4)) #passe d'un (6,2) à (3,4)

H=np.array([1,2,3])
H.shape #affiche (3,)
H=H.reshape((H.shape[0],1))
H.shape #affiche (3,1)
H=H.squeeze()
H.shape #affiche (3,)

I=A.ravel() #affiche [0. 0. 0. 0. 0. 0.]

print("==EXO==")
def initalisation(m,n):
	#m nb lignes
	#n nb colones
	#retourne matrice aléa (m, n+1)
	#colone tout à droite remplie de 1
	#indice : reshape, concatenate, randam.randn()

	#on créer matrice mxn aléatoire
	X = np.random.randn(m, n)
	#on créer colone de 1 avec même nb ligne que X
	#shape = (X.shape[0],1) car unique colonne
	Col = np.ones((X.shape[0],1))
	#on la concatene verticale avec colone
	X = np.concatenate((X, Col), axis=1)
	
	return X
#print(initalisation(2,2))


print("=====NUMPY INDEXING SLICING SUBSETTING======")
"""
Idée:naviguer dans les tableaux avec les techniques vues.
Comme liste, mais ici galère car espace à n dimensions 
En DataSciences, souvent tableaux 2D
Axe 0 = lignes et Axe 1 = colonnes
Pour pas perdre, naviguer seulement sur un axe à la fois; une seule donnée change

"""

#INDEXING
"""
Donc se déplacer avec INDEXING;

A[ligne, colonne]
depl vers la droite A[0,1]
depl vers le bas A[1,0]

"""
A=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(A[0,0]) #affiche 1


#SLICING
"""
#bases
A[début:fin(lignes), début:fin(colonnes)]

toute la colone n = A[:,n]
toute la ligne n = A[n,:]
mais ligne n s'affiche aussi A[n]

séléctionner un bloc : 
A[0:n, 0:n] de 0 à n-1 lignes et de 0 à n-1 colonnes
A[1:3, 1:] affiche [[_ _ _] lignes 1 et 2
				    [_ * *] colonnes 1 à la fin
				    [_ * *]]

mais si on connait pas fin du tableu : 
dernier:-1 av-dernier:-2
A[-1] : dernière ligne
A[:, -2:] = ttes lignes , colonnes à partie de avant-dernière

Séctionner un bloc pour lui donner une valeur:
A[0:2, 0:2] = 10
A maintenant égal à [[10,10,3][10,10,6][7,8,9]]

#intégrer le pas
A[début:fin:pas(lignes), début:fin:pas(colonnes)]
Ex : séléctionner ces données : 
								[[* _ * _ *]
							     [_ _ _ _ _] 
							     [* _ * _ *]
							     [_ _ _ _ _]
							     [* _ * _ *]]
A[::2, ::2]



"""
A[:,0] #affiche [1,4,7]
A[0,:] #affiche [1,2,3] 
A[0] #affiche [1,2,3]
A[0:2, 0:2] #affiche [[1,2][4,5]]
A[-2] #affiche avant dernière ligne
A[1:3, 1:] #affiche [[5,6][8,9]]

B=np.zeros((4,4)) #B=mat 4x4 de 0
B[1:3, 1:3] = 1 #B=mat 4x4 de 0 mais carré de 1 au centre
 
C=np.zeros((5,5)) #C=mat 5x5 de 0 
C[::2,::2]=1 #1 placé sur ttes lignes et colonnes avec pas de 2


#BOOLEAN INDEXING
"""
On peut obtenir un tableau de booléen avec une condition :
A tableau 5x5 d'entiers de 0 à 9
A<5 renvoi un tableau de booléen true ou false
Cela s'appelle un MASQUE
Utile pour images, filter données d'un tableau de meme dim

#Attention, A[A<5] = tab 1D des valeurs 
#			A(A<5) = tab 2D des booléens
"""

A=np.random.randint(0, 10, [5,5])#tab de 5x5 avec aléa de 0 à 9
A<5 #tableau de booleen sur cette condition
A[(A<5) & (A>2)]=10 #remplacer valeurs par 10 si conditions vérifiées

#Ex du masque avec image:
Pixels=np.random.randint(0, 255, [1024,720])
Pixels[Pixels>200]=255 #pixels avec luminosité élevée sont égaux à 255
#Ex du filtrage donnant une liste (tab 1D)
A[A<5] #tableau 1D des données de A accordant la condition

#Attention
B=np.random.randn(5,5)
print(B[A<5])#tableau 1D 
#données viennent de B mais séléctionnés par Booléen indexing de A


print("==EXO==")
"""
zoomer de 0,75 sur le centre de l'image puis appliquer un filtre dessus
en augmentant luminosité ou baissant (>200 ou <200)
photo chargeable directzement depuis module scipy
"""

#Charger le photo
"""
face est la photo = tableau numpy (verif avec type(face))
face = misc.face()
plt.imshow(face)
plt.show() #afficher image

#si charger en noir et blanc:
face = misc.face(gray=True)
plt.imshow(face, cmap=plt.cm.gray)
plt.show() #afficher image

face.shape #affiche (768,1024,3) => tab à 3D car 3 couleurs

pr exo, charge en noir et blanc, dc en 1D

from scipy import misc
import matplotlib.pyplot as plt
face = misc.face(gray=True)
plt.imshow(face, cmap=plt.cm.gray)
plt.show() #afficher image
"""
#zoomer
"""
utiliser dimensions avec le reshape et enregistrer
dimensions ds des var et les utiliser dans techinique
de slicing
"""
#print(face.shape)affiche (768,1024)
#zoom de 0,75==>> (576,768)

#CODE#
#importer imageet la mettre en noir et blanc
from scipy import misc
import matplotlib.pyplot as plt
face = misc.face(gray=True)

#Zoomer
print(face.shape)#affiche (768,1024)
#zoom de 0,75==>> (576,768)
#(768-576)/2 = 96 et (1024-768)/2 = 128
face=face[96:672, 128:896]

#Masquer avec booléen indexing
face[face>170] = 255
face[face<170] = 0

#print(face)

#Afficher image zoomée et masquée
plt.imshow(face, cmap=plt.cm.gray)
#plt.show() #afficher image



print("==========NUMPY STATS AND MATHS=============")
"""
faire des stats et maths 
import numpy as np
"""
print("=====MATHEMATIQUES======")

#METHODES NDARRAY
"""
sum() = faire somme des éléments
cumsum() = renvoi tab de somme cummulée
prod() = produit des éléments du tableau
cumprod() = produit cumulé
sort() = trier tableau
argsort() = renvoi tab des index du tableau trié
max() = renvoi max
argmax() = renvoi index de position du max
min() = renvoi min
argmin() = renvoi index de position du min
mean() = renvoi moyenne
... bcp autres

Appliquer methode sur les axe: 0(vertical) et 1(hori)
ex avec sum
sum(axis=0):tab(dim=colonnes)avec somme de chaque colonne
sum(axis=1):tab(dim=lignes) avec somme de chaque ligne

"""
np.random.seed(0) #fixer le random
A = np.random.randint(0,10,[2,3])#A=[[5,0,3], [3,7,9]]
A.sum() #affiche 27
A.sum(axis=1) #affiche colonne [8 19]
A.sum(axis=0) #affiche ligne [8 7 12]
A.argmin(axis=0) #affiche [1 0 0]
A.argsort()

#METHODES MATHS NORMALES
"""
sin()
exp()
...bcp autres

s'applique comme ça : np.exp(A) = renvoi tab avec valeurs expo

"""
np.exp(A)



print("=====STATISTIQUES======")


#METHODES NDARRAY
"""
mean() = moyenne du tab
var() = variance du tab
std() = ecart type du tableau
...bcp autres

"""
A.mean() #affiche 4.5

#METHODES STATS NORMALES
"""
Cours stats:
Coefficicent de corrélation linéaire R :
- nb qui permet de calculer l'intensité d'un lien linéaire entre deux variables
- R sans unité, situé entre -1 et 1
- R>0;lien linéaire positif (si X aug, Y aug) et R<0 ; lien linéaire négatif (si X aug, Y dim)
- plus R proche de 1 ou -1, plus lien fort
Droite régression:
- droite minimisant somme des carrés des distances verticlaes entre les points et elle même

Matrice de corrélation:
- mat qui indique les liens de coorélation entre lignes ou colonnes du tableau
- mat est symétrique et sa diago est constitué de 1 

np.corrcoef(A) #trace mat de corrolation des lignes du tableau A
Ex : [[5 0 3]		 	[[1, -0.53]
	  [3 7 9]]  donne 	 [-0.53, 1]] 
1 : car corrélation entre L1 et L1, donne 1
corrélation entre L1 et L2, donne -0.53
corrélation entre L2 et L1, donne -0.53
1 : car corrélation entre L2 et L2, donne 1

np.unique(A) #renvoi tableau 1D de toutes les données triées
np.unique(A, return_counts=True) #pareil que précéd. mais renvoi un 
second tableau 1D avec la répétition de chacune des données
Ex avec des binaires
A = np.random.randint(0,2,[7,7]) #mat de 0 et 1
print(np.unique(A, return_counts=True)) #affiche ([0, 1], [21, 28]




...bcp autres

"""
#print(A)

np.corrcoef(A) #mat corrélation 
np.corrcoef(A)[0,1] #affiche élément ligne 0 colonne 1 de mat corrélation
np.unique(A) #affiche [0 3 5 7 9]
np.unique(A, return_counts=True) #affiche ([0, 3, 5, 7, 9],[1, 2, 1, 1, 1])
values, count = np.unique(A, return_counts = True)

for i,j in zip(values[count.argsort()], count[count.argsort()]):
	print(f'Le nombre {i} apparailt {j} fois')



print("======NUMBER CORRECTIONS======")
"""
Quand on a pas des nombres, on doit filtrer ces données pour les
manipuler.
nan = not a number
Si calcule moyenne mean() sur un tableau avec des nan, marche pas

Surtout avec Panda quer nous verrons cela ! 
Mais il y a sur numpy qques methodes qui travaillent avec les nans
"""

V = np.random.randn(5,5)
V[0, 0] = np.nan # 0,0 = nan
V.mean() #affiche nan = mache pas
np.nanmean(V) #affiche la moyenne sans tenir compte de nan
np.std(V) #affiche ecart type sans tenir compte de nan
np.isnan(V) #renvoie tableau bool True si nan
np.isnan(V).sum() #renvoie nb de nan
np.isnan(V).sum()/V.size #renvoie % de nan dans tableau
V[np.isnan(V)] = 0 #passe à 0 tous les nan 

print("======ALGEBRE LINEAIRE======")

A = np.ones((2,3)) #mat 2l 3c
B = np.ones((3,2)) #mat 3l 2c

A.T #transposée de matrice A (échanger lignes et colonnes)
A.dot(B) #produit matriciel A.B = (2,3)*(3,2) = (2,2)
B.dot(A) #produit matriciel B.A = (3,2)*(2,3) = (3,3)

"""
package spécifique pour algèbre linéaire : 
numpy.linalg

"""
A = np.random.randint(0,10,[3,3])

np.linalg.det(A) #calucl determinant
np.linalg.inv(A) #inverse la matrice A
np.linalg.pinv(A) #inverse la matrice A sans passer par combi linéaires
np.linalg.eig(A) #renvoi tableau des valeurs propres et 2nd tableau des vecteurs propres


#EXO#
"""
Standardiser un dataset très utilisé ! 
Permet de mettre sur une même échelle toutes les diff colonnes/lignes
Formule : (Tableau - moy(Tableau))/ecarttype
Chaque tableau corresspond à une colonne

Standardiser une mat = toutes colonnes standardisées : 
veut dire que la moyenne de chaque colonne est égal à 0 and std = 1
Et dc chaque colonne suit une distribution normale parfaite
Le fait d'avoir une mat standardiser facilite les statistiques.

Standardiser la matrice A sur chaque colonne
"""
np.random.seed(0)
A = np.random.randint(0,100,[10,5])
print(A)
#il faut donc travailler sur axe 0
#On fait (10,5) - (5,)(broadcasting)
D = (A-A.mean(axis=0)) / A.std(axis = 0)
D.mean(axis = 0) # = 0
D.std(axis = 0) # = 1



print("============BROADCASTING============")
"""
Technique très puissante mais dangereuse : Broadcasting

Technique pas obligatoire, mais plus une façon dont Numpy a été conçue.
Permet de faire calculs entre tab A et tab B très rapidement.
Ex : en C, pour faire en sorte que A[i,j] + B[i,j] = C [A[i,j]+B[i,j]]
Double boucle for,... long

Avec Numpy juste C = A+B
Mais cela ne concatene pas, les valeurs a(i,j) et b(i,j) s'ajoutent
Cela ne marche que avec des matrices NUMPY de MEME SHAPE

"""
A = np.random.randint(0,10, [3,3])
B = np.random.randint(0,10, [3,3])
#K1+K2 seukement si K1.shape == K2.shape

"""
Mais il existe une technique pour "etendre les tableaux" afin de 
s'additionner à tous les autres tableaux : 
Le BROADCASTING = étendre les dimensions d'un tableau

Mais ne s'applique que si matrices B à une seule colonne et meme 
nb de lignes. (ex : (2,3)+(2,1) = OK mais (2,3)+(2,2) = !OK)
>Mais :
Seulement, une unique colonne (n,1) et une unique ligne (1,m)
puvent se broadcaster et former une matrice (n,m) !!
Mais conséquences désastreuses car peut former des tableaux à 
grandes dimensions non vouls ! 
Donc TOUJOURS vérifier les dimensions de matrices avec .shape, et 
les redimmensionner avec .reshape()
"""
A = np.random.randint(0,10, [4,1]) #une colonne n = 4
B = np.random.randint(0,10, [1,3]) #une ligne m=3
A+B #donne matrice (4,3)


print("============RESUME NUMPY============")
"""
- Utilisé pour travailler sur tableaux à n dimensions
(en M L surtout à 2 dimlensions )

- Axe 0 : vertical et Axe 1 : horizontal (important car utile partout)

- Attribut utiles : 
shape (rend nb lignes et nb colonnes du tab)
size (nb éléments)

- methodes : 
concatenate(axis) (concatener avec axe 1 ou 0)
reshape((shape)) (redimensionner un tab)
ravel() (applatir tab à une seule dimension)

argsort(axis) (classement, trie dans le tableau)

- boolean indexing : 
A[A<10] (très utile)

- constructeurs : 
np.array(objet, dtype)
np.zeros((shape), dtype)
np.ones((shape), dtype)
np.random.randn(lignes, colonnes)
np.random.randint(min, max, (shape))

shape en []
"""