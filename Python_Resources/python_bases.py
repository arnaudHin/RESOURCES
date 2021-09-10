#AUTOR : Arnaud HINCELIN
#Date : 07/2020


En python ==>> Indentation compte !!!!!!!!!!!!  (vs tous autres langages)

#====================================================
#FONCTIONS ET VARIABLES
#====================================================

#Fonction simple avec lambda
f = lambda x: x**2
a=f(2)
#print(a)

d = lambda x,y:x*2+y
#print(d(1,1))

#fonctions avancées avec def
def e_pontentielle(masse, hauteur, g=9.81, e_limit=3000):
	E = masse*hauteur*g
	if E>e_limit:
		E=e_limit
	return E

#print(e_pontentielle(80,5), f(4))

def var(x):
	if (x>0):
		return "petit positif"
	elif x==0:
		return "nul"
	else:
		return "negatif"


#====================================================
#BOUCLES
#====================================================

for element in range(3):
	print(element)
for l in range(5,10):
	print(l)
for element in range(5,-5,2):
	print(element)

for i in range(5,-5, -2):
	print(i, var(i))

x=1
while x<=10:
	#print("yeah")
	x+=1


#====================================================
# SEQUENCES
#====================================================
"""
Ensemble d'éléments ordonnés, qui suivent un ordre.
Chaque élément a un index (commence à index 0 à n-1)
"""
#INDEXING : acceder à un élément depuis un index
"""
index 0 : premier élément
index 1 : second élément
index n-1 : dernier élément
index -1 : dernier élément
index -2 : avant dernier élément
...
"""

#SLICING : acceder à une fourchette d'éléments depuis des index
#index début, index fin, index pas (par défaut 1)
"""

[0:3] = [:3] = premier jusqu'au troisième
[3:-1] = [3:] = qutrième jusqu'au dernier
[0:-1:2] = [::2] = premier, troisième,...
[0:-1:-1] = [::-1] = inverse la séquence

pour une liste ou tuple; entre crochets 
nom_sequence[debut, fin, pas]
"""

#Liste

liste_1 = [1,3,5,3,8]
villes = ['Paris', 'Londres', 'Berlin']
liste_2 = [liste_1, villes]
liste_3 = []

print(villes)
print(villes[2])
print(villes[-1])
print(liste_1[:4])


#Tuple
"""
Comme listes mais non modifiables.
Utile pour rentrer des données protégées, et adaptées 
si grosse quantité de données (rapide, peu mémoire,...)
"""
tuple_1 = (1,4,8,9,0)
tuple_ville = ('Paris', "Londres", 'Berlin', 'NewYork')

print(tuple_1)
print(tuple_ville[1])
print(tuple_1[1:])
print(tuple_ville[::2])
print(tuple_ville[::-1])

#String
prenom = 'Arnaud'
print(prenom[1])
print(prenom[::-1])

#Méthodes à appliquer sur Listes
"""
sur n'importe quel séquence, on peut appliquer des méthodes
nom_sequence.method

.append(10) #ajouter élément à la fin
.append('Madrid')

.insert(1, 20) #insérer élément à index 1

.extend(tuple_1) #ajouter une séquence à la fin

len(liste_1) #renvoi taille = index fin + 1

.sort() #trie en ordre alphabetique ou croissant
.sort(reverse=True) #trie en ordre non alphabetique ou décroissant

.count('Paris') #renvoi le nombre de fois que element présent

"""
#Conditions/Boucles + Listes


if 'Paris' in villes: #vérifier présence element 
	print ('présent')
else:
	print('non present')


for valeur in villes: #afficher valeurs
	print(valeur)

for index,valeur in enumerate(villes): #afficher index et valeurs
	print(index, valeur)

for a,b in zip(villes, liste_1): #affiche element associés au même index de deux listes
	print(a, b)



#====================================================
# DICTIONNAIRE
#====================================================
"""
Ensemble d'affectations clé-valeur (tout type)
dic={
	clé:valeur
}
Dico n'a pas d'ordre, pas d'index

"""

dictionnaire_1 = {
	'chien':'dog',
	'chat':'cat',
	'souris':'mouse',
	'oiseau':'bird'
}

dictionnaire_2 = {
	'Arnaud' : 20,
	'Louis' : 21,
	'Charles' : 12
}

dictionnaire_3 = {
	'cle_1' : dictionnaire_1, #contient un dico
	'cle2' : dictionnaire_2
}
 
print(dictionnaire_1)

#Methode à appliquer sur le dico

.values() #renvoi liste des valeurs
.keys() #renvoi liste des clés
len(dict) #renvoi taille de dico
dict['cle'] = valeur #ajout couple en fin de dict
.get('cle non comprise', valeur) #renvoi valeur pour une clé nn comprise

#créer dico à partir de liste, qui seront les clés
liste_tuple = ('Paris', 'Madrid', 'Dublin')
dico.fromkeys(liste_tuple, 'Default') #créé dico et associe à chaque clé la valeur 'Default'

dico.pop('cle') #enlève couple du dico, et retourne la valeur associée



#AVEC BOUCLES 

#Affiche les clés
for i in dictionnaire_2:
	print(i)

#Affiche les valeurs
for i in dictionnaire_2.values():
	print(i)

#Affiche les clés et valeurs
for i,j in dictionnaire_2.items():
	print(i,j)


#====================================================
# TECHNIQUE : LIST/DICT COMPREHENSION
#====================================================


#LIST COMPREHENSION
"""
insérer la boucle for à intérieur de la liste.
C'est plus court, plus pro, moins mémoire utilisé et calcul plus rapide.
"""
#Au lieu de :

maList=[]
for i in range (10):
	maList.append(i**2)

# On utilise :
maList2 = [(i**2) for i in range (10)]
print(maList)
print(maList2)
#rapide peut se prouver en utilisant le temps : 
import time
start = time.time()
...
end = time.time()

#créer une Nested List : Listes dans une Liste
maList3 = [[i for i in range(3)] for j in range(3)]
print(maList3) #affiche [0,1,2],[0,1,2],[0,1,2]


#DICO COMPREHENSION
"""
créer des dicos à partir de listes
"""
prenoms = ['Arnaud', 'Louis', 'Charles']
dico_comp = {k:v for k,v in enumerate(prenoms)}
# affiche {0: 'Arnaud', 1: 'Louis', 2: 'Charles'}


prenoms2 = ['Arnaud', 'Louis', 'Charles']
age = [20, 21, 12]
dico_comp2 = {prenoms2:age for prenoms2,age in zip(prenoms2, age)}
#Affiche {'Arnaud': 20, 'Louis': 21, 'Charles': 12}

#POUR LES DEUX : Inclure conditions après les for
dico_comp3 = {prenoms2:age for prenoms2,age in zip(prenoms2, age)if age > 19}
# Affiche {'Arnaud': 20, 'Louis': 21}

#RESUME DES DICT/LIST COMPREHENSION

"""
3 parties : 
Fait ca   Pour la collection     Dans cette situation
[x**2     for i in range(0, 50)  if(x%3 == 0)]

"""

#TUPLE COMPREHENSION
"""
ATTENTION, ne se fait pas de la manière
On ne peut faire :
"""
tuple_comp = (i**2 for i in range(10))
#Affiche <generator object <genexpr> at 0xb7799d1c>
#On obtient un générateur
tuple_comp2 = tuple((i**2 for i in range(10)))
#OK


#====================================================
# FONCTIONS BUILT-IN
#====================================================


abs(x) #valuer absolue
round(x) #arrondi
max(maListEx) #max d'une liste
min(maListEx) #min d'un liste
len(maListEx) #taille d'une liste
sum(maListEx) #renvoi somme des élements de liste
all(maListEx) #renvoi True si tous élements sont True
any(maListEx) #renvoi True si au moins 1 élement est True

type(x) #renvoi type
str(x) #convertir en String
int(x) #convertir en Integer
float(x) #convertir en Float
dict(maListEx) #convertir en dico (marche avec list, tuple)
tuple(maListEx) #convertir en tuple (marche avec liste et dico)
list(monDico.keys()) #onvertir en une liste les clés

bin(x) #convertit int en binaire
hex(x) #convertir int en hexadecimal
bytes(x) #convertir int en byte
oct(x) #convertir int en octet

input('entrer un nb') #saisir du text renvoi String


#fonction format()

"""
permet de personnaliser des chaines de cara, 
de les rendre dynamique. Surtout pour intégrer des variables dedans
Ex : 
x=5
typ='lebel'
print("Vous avez x fusils de type typ") #affiche pas les variables

Solution : remplacer var par {} et placer un .format(var) à la fin
"""
x=5
typ='lebel'
print("Vous avez {} fusils de type {}".format(x,typ))
#ou
print(f"Vous avez {x} fusils de type {typ}")

#Ex utilsé pour accéder à certaines clés de dico: 
reseaux={
	"W1":3,
	"b1":7,
	"W2":0,
	"b2":6,
	"W3":90
}

for i in range(1,4):
	print("couche",i)
	print(reseaux[f"W{i}"])
#Affiche seulement les valeurs associées aux clés W1,W2 et W3

#fonction open()

'''
Permet d'ouvrir ou créer des fichiers.

modes : 
r : ouvrir pour lire un fichier déjà existant
w : ouvrir pour écrire (ou créer un fichier)
a : ouvrir pour érire à la fin
'''
f = open('fichier.txt','w')#créer fichier 
f.write('bonjour oui !')#écrire
f.close() #fermer fichier

f = open('fichier.txt','r')#créer fichier 
print(f.read())
f.close()

#pour faire plus court et sans close (pareil pour lire mais 'r'): 
with open('fichier.txt', 'w') as f:
	f.write("Hey ! comment vas tu ?")





#====================================================
# MODULE ET PACKAGES
#====================================================
"""
un fichier avec un progamme est appelé module.
Si dans un autre fichier on a besoin du programme,
on peut importer le fichier avec :
import fichier_1

pour appeller les fonctions:
fichier_1.fonction()
"""
#Ex :
import Exo
Exo.SuiteFibo(3)

#rmq : un dossier __pycache__ est créé, gère les connexions

#on peut donner une surnom à notre module importé :
import Exo as fichier_1
fichier_1.SuiteFibo(5)

#on peut importer seulement certaines fonctions/variables 
from Exo import SuiteFibo
SuiteFibo(5)

#on peut tout importer : 
from Exo import *

"""
Un package est paquet de plusieurs modules.
"""

##MODULES UTILES EN DATA SCIENCES##

#math : toutes fonctions nécessaires pour les maths
import math
math.cos(180)
math.pi

#statistics : toutes fonctions stats nécessaires
import statistics
statistics.mean(liste_1) #moyenne de liste
statistics.variance(liste_1)


#random : générer nombre aléatoire
import random
#random.seed(0) #permet de régler l'aléatoire au même résultat
random.choice(liste_1)#choisi élement de liste
random.choice(['julie', 'arnaud', 'mister'])#idem
random.random()#générer float compris entre 0 et 1
random.randint(5,10)#entier entre 5 et 10
random.randrange(100)#entier entre 0 et 100
random.sample(range(100), 10) #liste de 10 entier de 0 à 100 
random.sample(range(100), random.randrange(10)) #liste d' entier de 0 à 100 dont nombre est aléatoirebentre 0 et 1à
random.shuffle(['julie', 'arnaud', 'mister']) #mélanger structure de données
random.shuffle(liste_1) #mélanger structure de données


#os : 
import os

os.getcwd() #renvoi le directory actuel


#glob : 
import glob

glob.glob("*")#renvoi liste de tous les noms des fichiers du répertoire
glob.glob("*.txt")#renvoi liste de tous les noms des fichiers .txt du répertoire





#====================================================
# POO
#====================================================
"""
Considérer que univers modélisé par 3 notions :
Objets
Attributs
Actions

Pour construire un objet, on utilise un constructeur (similitude avec java)
On peut appliquer les méthodes seulement si objet créé

En machine learning, pas important de savoir crééer des classes,
on va seulement utiliser des objets de certaines classes.

"""
import numpy as np

#créer un objet de classe :
#numpy.ndarray() : tableau à n dimensions
#selon doc, créer un objet avec np.array([1,2,3])
tableau_2 = np.array([1,2,3])

#on peut ensuite appliquer à objet nos méthodes et nos attributs
tableau_2.size #attribut size
tableau_2.sort() #methode sort() pour tirer


#====================================================
# CLASSES
#====================================================
"""
En POO, une classe est un objet.
On utilise le mot clé self pour désigner l'objet lui même au sein de la classe.
Un classe est caractérisée par des attributs (caractéristiques de l'objet) et 
des fonctions (actions possibles pour l'objet)
"""
#self = this en python par rapport au java

#Ex de classe : 
personnage(self, nom, vie):
	#attribut
	self.nom = nom
	self.vie  = vie

	#methodes
	def avancer():
		gtg






#====================================================
# libs - SEB
#====================================================
"""
#mathplotlib
besoin de représenter graphiquement notre programme
Utilisé pour la visualisation de l'allure, prédiction !
Plus besoin une fois le programme fini !

#numpy
maths

#matlab
matrices
"""
#====================================================
# IMAGE 
#====================================================
"""
matrices pas pratiques à utiliser, mais des vecteurs :
mise à la suite pour former une liste
matrice de 16x20 ==>> vecteur de 3200

Vérifier qu'un objet observé est bien celui que l'on prédit :
proba = loi normale, 
valeurs entre

Sigmoid (numpy)
valeurs entre 0 et 1


#IA pour image:
recherche de pattern (diffs entre photos de toi et autre)


"""

Liste = [2,4,4,5,3,4,2,7,7,8,5,5,1]

def rechercheElement(data, liste):
	result = False
	for i in liste:
		if(i==data):
			result = True
	return result

print(rechercheElement(7, Liste))

def unique(liste):
	n = len(liste)
	unique = [None for i in range(n)] 
	unique[0] = liste[0]
	for i in range(1,n):
		#print(unique)
		print(liste[i])
		print(rechercheElement(liste[i], unique))
		if(rechercheElement(liste[i], unique) == False):
			unique[i] = liste[i]
	return unique

print(unique(Liste))	


