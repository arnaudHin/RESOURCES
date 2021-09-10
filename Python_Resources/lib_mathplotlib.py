#AUTOR : Arnaud HINCELIN
#Date : 07/2020

print("==============MATHPLOTLIB==================")
"""
Sert à tracer tous les graphiques qui existent
Graphics servent à voir les choses sur lesquelles on travaille (données,...)
buts : 
- Mieux comprendre le pb
- Mieux expliquer le phénomène
- Voir, plutôt qu'imaginer (l'abstrait)
- Nous aider

Mais très souvent, bugs et problèmes avec mathplotlib.
Donc gens passent temps à les résoudre ! 
Pourtant super simple à utiliser : 
- éviter de rajouter trop de détails dans les courbes
- Attention à ne pas mélanger les 2 méthodes : 
	- méthode OOP (orienté objet)
	- plt.plot ("normal"= Mathplotlib Fonction)
"""
print("=======METHODE 1 : MATHPLOTLIB FONCTION=======")
"""
2 méthodes pour créer graphs (même résultat)
Fonction : 
Plus simple, on utilise une fonction appeléee plot 

#FUNCTION

Il faut importer le module matplotlib.pyplot
import matplotlib.pyplot as plt

- créer fonction et entrer en param les données à afficher axe X et en Y
plt.plot(X,Y)
On peut créer plusieurs fonctions (plt.plot(x2,Y2),...)

- créer nuage de points 
plt.scatter(x,y)

Il faut ensuite afficher avec (si plusieurs fonctions, couleur diff) 
plt.show()

Attention, erreur fréquente : passer en param tableaux n'ayant pas
mêmes DIMENSIONS !



"""
import numpy as np #import numpy
import matplotlib.pyplot as plt #import pyplot
x = np.linspace(0,2,10) #tableau numpy 1D 10 chiffres de 0 à 1
y = x**2 #tableau des carrées de x
#plt.scatter(x,y) #nuage de points
#plt.plot(x,y) #fonction

#plt.show() #afficher le tout 

#Bien d'autres graphs en 3D,... vus plus tard ! 



#STYLE
"""
Customiser une courbe, infinité de possibilités ! 
Problème... faire au plus simple
paramètres : 
- c : couleur 
'red', 'black', 'blue',...
- lw : épaisseur de courbe
1,2,3,...
- ls : style de courbe
'--', '..',...
- label : nom de courbe
'voltage',...

plot(x,y, label='Nom', lw=3, ls="-", c='red')
"""
#plt.plot(x,y,label = "Nom de courbe", lw = 3, ls = '--', c='red')
#plt.show()


#CYCLE DE VIE
"""
Quand ecrit plt.plot(), on a créé la figure
Mais en principe, on commence par créer figure vide, la fenetre de travail
-> plt.figure()
Nous pouvons définir taille en paramètre
plt.figure(figsize = (12,8))

puis on créer la courbe avec plt.plot(x,y)
-> plt.plot(x,y)
On peut créer plusieurs courbes 
-> plt.plot(x,x**3)

On peut rajouter titre de graphique / nom des axes / nom des courbes
-> plt.title('Courbe du gain')
-> plt.xlabel('axe x')
-> plt.ylabel('axe y')
-> plt.legend() #affiche le nom de courbe rentré en param du plot (label)

On peut maintent afficher 
->plt.show()

Et enregistrer le graph si besoin dans répertoire du fichier .py
-> plt.savefig('graph1.png')

"""

#plt.figure()
#plt.plot(x,y,label = "fonction carré", lw = 3, ls = '--', c='red')
#plt.title('Courbe du carré')
#plt.xlabel('abscisses')
#plt.ylabel('ordonnées')
#plt.legend()
#plt.show()

#plt.savefig('graph1.png')


#SUBPLOT
"""
Souvent , afficher plusieurs graphs sur même fenetre.
Methode subplot() permet de générer une grille su notre fenetre
En paramètres, on rentre le nb de lignes et colones 

En fait, commencer par créer une fenetre 
-> plt.figure()
Puis implémenter une grille 2l et 2c et le graph 1 dan sla case 1
-> plt.subplot(2,2,1)
-> plt.title('One')
... contenu

Puis à la suite, faire le second graph2 dans case 3
-> plt.subplot(2,2,3)
-> plt.title('Two')
... contenu

Finir par un show

Rmq : si on veut plusieurs fenetre, avant chaque graph on appele figure()
"""

#plt.figure()
#plt.subplot(2,2,1)
#plt.plot(x,y)
#plt.title('One')

#plt.subplot(2,2,2)
#plt.plot(x,y*2)
#plt.title('Two')

#plt.subplot(2,2,4)
#plt.plot(x,x)
#plt.title('Three')

#plt.show()

print("=======METHODE 2 : OOP =======")
"""
Methode plus technique mais plus de fonctionnalitées.
Non recommandée...

Commencer par créer la figure et les axes qui sont des objets
fig, ax = plt.subplots()
rmq : si plusieurs graphs, créer grille et subplots(2,2,1)...

puis rentrer paramètres pour créer la fonction
ax.plot(x,y)

et afficher 
plt.show()

Simple...
OOP intérressant car plus de possibilités ; 
- créer graphs qui partagent la même abscisse ou ordonnée
	suffit d'écrire : 
	fig, ax = plt.subplots(2,1,sharex=True)
	mais cela renverra une erreur, car en fait ax n'est pas vraiment 
	un objet mais un tab numpy qui va contenir nos objets
	(se voit d'ailleurs avec type(as) qui donne numpy.array).
	ax est donc tableau de dimension 2 (ici)
	il faut écrire les fonctions plot à la suite avec cette notation
	ax[0].plot() fonction 1
	ax[1].plot() fonction 2
"""

#graph simple
#fig, ax = plt.subplots()#fenetre simple
#ax.plot(x,np.sin(x)) #fonction
#plt.show()

#deux graphs séparés mais ayant les mêmes abscisses
#fig, ax = plt.subplots(2,1, sharex = True) #fenetre avec grille
#ax[0].plot(x,y) #fonction 1
#ax[1].plot(x, np.sin(x)) #fonction2
#plt.show() # f1 et f2 vont partager même axe x



print("=======EXO 1=======")
"""
Dictionnaire qui possède 4 datasets (nuage de 100 points)
(clé = experience{i})
(valeurs = tab numpy de 100 points)

Créer fonctions graphique qui utilise ce dico et qui va créer une 
figure avec quatres graphiques qui corresspondent à une des expériences

"""
#créé à l'aide des dico comprehension
dataset = {f"experience{i}": np.random.randn(100) for i in range(4)}

def traceGraph(dataset):
	x = [i for i in range(100)]
	plt.figure(figsize = (8,12))
	for i in range(3):
		y = dataset.get(f'experience{i}')
		plt.subplot(4,1,i+1)
		plt.plot(x, y)
		plt.title(f"experience{i}")
	plt.show()

#traceGraph(dataset)



print("===========MATHPLOTLIB GRAPHS SPECIAUX===============")
"""
5 graphiques à tracer les plus cool
"""

#1 : Graph de classification avec plt.scatter (nuage de points)
"""
En machine learning, la moitié des pbs abordés sont des pbs de 
classification.

EX :  trie d'un mail en spam ou non en fonction du 
nb de fautes et de liens ;c'est une ia qui gère sa classification
Ici on a deux classes : SPAM et NONSPAM
et deux variables : fautes et liens
Graph plt.scatter = variables en axes (fautes = axeY et liens = axeX)
C'est le meilleur choix pour représenter notre dataset!! 

Mais on va travailler avec un dataset plus connu : fleurs d'Iris
Il contient 150 exemples de fleurs d'Iris répartis en 3 classes et 
on dispose de 4 variables pour prédire de quelle classe il s'agit.
Variables : longueur/largeur du petal et longueur/largeur du sepal.

pour travailler, besoin de packages :
numpy
mathplotlib.pyplot
load_iris (package = sklearn.datasets mais juste besoin de fonction)

installer les packages (numpy, scipy, joblib, et scikit-learn) : 
$ pip3 install -U scikit-learn

à l'intéreur de sklearn, plusieurs datasets dont fleurs d'iris.
et dont des données (nos 4 variables) et des targets (nos 3 classes)


Visualiser :
fonction scatter, et en paramètres, deux des 4 varaibles en abscisses
et ordonnées. 
x[:,0] => les 150 lignes et un index parmis les 4 (0 à 3)
0 = longuer sepal et 1 = largeur sepal
plt.scatter(x[:,0], x[:,1])

mais mieux encore : 
colorier les points suivant la nature de leur classe
rajouter argument : c=y (car y corresspond à chque classe 0,1 ou 2)

modifier transparance des points : alpha = 0.5

modifier taille des points : s=100
modifier taille en fonction de leur valeur d'une varibale: 
ex avec var 3 s=x[:,2]*100
plus gros, plus var 3 (petale long) est grande
plus petit, plus var 3 (petale long) est petite


Super, mais pb est que on ne peut repsésenter que 2 des varaibles
sur ce graphique

"""
#import numpy as np #import numpy  DEJA IMPORTE
#import matplotlib.pyplot as plt #import pyplot  DEJA IMPORTE
from sklearn.datasets import load_iris

iris = load_iris() #recupere le dataset
x = iris.data #tableau variables 150 lignes et 4 colonnes
y = iris.target #tableau classes 150 valeurs 0,1 ou 2 qui represente nos 3 classes
names = list(iris.target_names) #3 noms de chaque fleur

print(f'x contient {x.shape[0]} exemples et {x.shape[1]} variables \n')
print(f'il y a {np.unique(y).size} classes\n')

#plt.scatter(x[:,0], x[:,1]) #obtient nuage de points voulus
plt.scatter(x[:,0], x[:,1], c=y, alpha= 0.5, s=x[:,2]*100) #nuage de points triés par couleurs selon classe
plt.xlabel('longeur sepal') #a
plt.ylabel('largeur sepal')

#plt.show()



#2 : Graph 3D
"""
type vient npltoolkit, 
permet de créer graphs en 3D, utile pour visualiser plus de variables
sur un même graphique.



"""
from mpl_toolkits.mplot3d import Axes3D

ax = plt.axes(projection='3d')
f = lambda x,y: np.sin(x) + np.cos(x+y)




