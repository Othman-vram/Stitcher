# Interface de Réarrangement et de Suture Rigide de Fragments Tissulaires

Une application de bureau professionnelle pour visualiser, manipuler et suturer plusieurs fragments d'images tissulaires provenant de fichiers TIFF pyramidaux et SVS.

## Table des Matières

1. [Prérequis](#prérequis)
2. [Installation](#installation)
3. [Démarrage de l'Application](#démarrage-de-lapplication)
4. [Interface Utilisateur](#interface-utilisateur)
5. [Chargement des Images](#chargement-des-images)
6. [Manipulation des Fragments](#manipulation-des-fragments)
7. [Sélection de Groupe](#sélection-de-groupe)
8. [Points Étiquetés](#points-étiquetés)
9. [Suture Rigide](#suture-rigide)
10. [Exportation](#exportation)
11. [Raccourcis Clavier](#raccourcis-clavier)
12. [Dépannage](#dépannage)

## Prérequis

### Images Prétraitées Requises

**IMPORTANT :** Cette application est conçue pour fonctionner avec des images tissulaires **prétraitées et préparées** :

#### **Format d'Image Requis :**
- **TIFF pyramidal** avec structure multi-résolution
- **Images RGBA** (4 canaux : Rouge, Vert, Bleu, Alpha)
- **Fond transparent** (canal alpha = 0) pour les zones non-tissulaires
- **Résolution cohérente** entre tous les fragments

#### **Prétraitement Nécessaire :**

**1. Segmentation du Tissu :**
- Séparation du tissu du fond
- Suppression des artefacts et bulles d'air
- Masquage des zones non-pertinentes

**2. Normalisation :**
- Correction de l'illumination
- Normalisation des couleurs
- Ajustement du contraste

**3. Format de Sortie :**
- Conversion en TIFF pyramidal avec niveaux multiples
- Canal alpha correctement défini
- Métadonnées de résolution (microns par pixel)

#### **Structure Pyramidale Recommandée :**
```
Niveau 0: Résolution complète (ex: 40x, 0.25 µm/pixel)
Niveau 1: 1/2 résolution (ex: 20x, 0.5 µm/pixel)
Niveau 2: 1/4 résolution (ex: 10x, 1.0 µm/pixel)
Niveau 3: 1/8 résolution (ex: 5x, 2.0 µm/pixel)
...
```

#### **Outils de Prétraitement Recommandés :**
- **Scripts Python personnalisés** avec OpenCV et scikit-image
- **OpenSlide** : Manipulation de fichiers pyramidaux (déjà inclus dans les dépendances)

#### **Exemple de Pipeline de Prétraitement :**

1. **Chargement de l'image source** (TIFF haute résolution)
2. **Détection automatique du tissu** (seuillage, morphologie)
3. **Nettoyage du masque** (suppression des petits objets)
4. **Application du masque** (fond → transparent)
5. **Création de la pyramide** (downsampling successif)
6. **Export en TIFF pyramidal** avec compression LZW

### Configuration Système Requise

- **Système d'exploitation** : Windows 10+, macOS 10.14+, ou Linux (Ubuntu 18.04+)
- **RAM** : 8 Go minimum, 16 Go recommandés
- **Espace disque** : 1 Go d'espace libre
- **Carte graphique** : OpenGL 2.1 ou supérieur
- **Python** : Version 3.8 ou supérieure

### Dépendances Python Requises

L'application nécessite les packages Python suivants :

```
PyQt6==6.6.1                 # Framework d'interface graphique
opencv-python==4.8.1.78      # Traitement d'images
numpy==1.24.3                # Opérations numériques
Pillow==10.1.0                # Manipulation d'images
openslide-python==1.3.1       # Support des images pyramidales
scikit-image==0.22.0          # Algorithmes de traitement d'images
scipy==1.11.4                 # Calcul scientifique
matplotlib==3.8.2             # Visualisation
tifffile==2023.9.26           # Gestion des fichiers TIFF
```

### Bibliothèques Système Supplémentaires

#### Windows
- **OpenSlide** : Téléchargez et installez depuis [openslide.org](https://openslide.org/download/)
- **Microsoft Visual C++ Redistributable** : Requis pour OpenCV

#### macOS
```bash
brew install openslide
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install openslide-tools
sudo apt-get install python3-openslide
sudo apt-get install libgl1-mesa-glx  # Pour OpenGL
```

## Installation

### 1. Cloner ou Télécharger le Projet

```bash
git clone <url-du-projet>
cd tissue-fragment-stitching
```

### 2. Créer un Environnement Virtuel (Recommandé)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Installer les Dépendances

```bash
pip install -r requirements.txt
```

### 4. Vérifier l'Installation

```bash
python -c "import PyQt6, cv2, numpy, openslide; print('Installation réussie!')"
```

## Démarrage de l'Application

### Lancement Standard

```bash
python main.py
```

### Lancement avec Débogage

```bash
python main.py --debug
```

L'application devrait s'ouvrir avec une interface sombre optimisée pour l'imagerie médicale.

## Interface Utilisateur

### Vue d'Ensemble

L'interface est divisée en plusieurs zones principales :

1. **Barre d'Outils** (en haut à gauche) : Actions principales
2. **Liste des Fragments** (à gauche) : Gestion des fragments chargés
3. **Panneau de Contrôle** (à gauche, en bas) : Manipulation des fragments
4. **Canevas Principal** (centre/droite) : Visualisation et interaction
5. **Barre de Statut** (en bas) : Informations et progression

### Barre d'Outils

La barre d'outils contient les actions principales :

- **📁 Charger Images** : Ouvre le dialogue de sélection de fichiers
- **💾 Exporter** : Exporte l'image composite et les métadonnées
- **🔗 Suture Rigide** : Lance l'algorithme de suture automatique
- **🔄 Réinitialiser** : Remet toutes les transformations à zéro
- **🗑️ Supprimer** : Supprime le fragment sélectionné

### Liste des Fragments

Affiche tous les fragments chargés avec :
- **Case à cocher** : Contrôle la visibilité
- **Miniature** : Aperçu du fragment
- **Nom** : Nom du fichier
- **Dimensions** : Taille en pixels
- **Bouton ×** : Suppression rapide

### Panneau de Contrôle

#### Onglet Fragment (Sélection Simple)

**Informations du Fragment :**
- Nom du fragment sélectionné
- Dimensions originales

**Transformations :**
- **↺ 90°** / **↻ 90°** : Rotation par pas de 90°
- **Angle personnalisé** : Champ de saisie pour rotation libre
- **Boutons rapides** : 45°, 135°, 225°, 315°
- **Ajustement fin** : +1° / -1° pour rotation précise
- **↔ Horizontal** / **↕ Vertical** : Retournement

**Position :**
- **↑ ↓ ← →** : Déplacement par pas de 10 pixels

#### Onglet Groupe (Sélection Multiple)

**Rotation de Groupe :**
- **↺ 90° CCW** : Rotation antihoraire du groupe
- **↻ 90° CW** : Rotation horaire du groupe
- **Rotation personnalisée** : Champ d'angle pour rotation libre du groupe
- **Boutons rapides** : 45°, 135°, 180°, 270°

**Déplacement de Groupe :**
- **↑ ↓ ← →** : Déplacement du groupe entier

## Chargement des Images

### Formats Supportés

L'application est conçue pour fonctionner avec :
- **TIFF/TIF** : Images TIFF pyramidales prétraitées

**Note importante :** Bien que le code supporte techniquement d'autres formats (PNG, JPEG), l'application est optimisée pour des fichiers TIFF pyramidaux prétraités avec canal alpha.

### Procédure de Chargement

1. **Cliquez sur "📁 Charger Images"** ou utilisez `Ctrl+O`
2. **Sélectionnez un ou plusieurs fichiers** dans le dialogue
3. **Attendez le chargement** (barre de progression visible)
4. **Les fragments apparaissent** dans la liste et sur le canevas

### Gestion des Images Pyramidales

Pour les fichiers TIFF pyramidaux :
- L'application charge automatiquement le niveau approprié
- Les niveaux de résolution sont gérés automatiquement
- L'exportation peut préserver la structure pyramidale

## Manipulation des Fragments

### Sélection d'un Fragment

**Méthodes de sélection :**
- **Clic sur le canevas** : Sélectionne le fragment sous le curseur
- **Clic dans la liste** : Sélectionne le fragment correspondant

**Indication visuelle :**
- Contour bleu autour du fragment sélectionné
- Mise en surbrillance dans la liste

### Déplacement

**Déplacement à la souris :**
1. Cliquez et maintenez sur un fragment
2. Glissez vers la nouvelle position
3. Relâchez pour confirmer

**Déplacement précis :**
- Utilisez les boutons fléchés du panneau de contrôle
- Chaque clic déplace de 10 pixels

### Rotation

**Rotation par pas de 90° :**
- **↺ 90°** : Rotation antihoraire
- **↻ 90°** : Rotation horaire

**Rotation libre :**
1. **Saisissez un angle** dans le champ "Angle" (0-360°)
2. **Cliquez "Appliquer"** ou appuyez sur Entrée
3. **Utilisez les boutons rapides** : 45°, 135°, 225°, 315°
4. **Ajustement fin** : +1° et -1° pour rotation précise

**Exemples d'angles courants :**
- **45°** : Rotation diagonale
- **135°** : Rotation diagonale inverse
- **180°** : Retournement complet
- **270°** : Équivalent à -90°

**Rotation libre (angles personnalisés) :**
- **Champ de saisie d'angle** : Entrez n'importe quel angle (0-360°)
- **Boutons prédéfinis** : 45°, 90°, 180°, 270°
- **Rotation fine** : Boutons +1° et -1° pour ajustements précis

**Comportement :**
- La rotation se fait autour du centre du fragment
- L'image est recalculée automatiquement
- La position peut être ajustée après rotation

### Retournement

**Types de retournement :**
- **↔ Horizontal** : Miroir horizontal
- **↕ Vertical** : Miroir vertical

**Combinaisons possibles :**
- Horizontal seul
- Vertical seul
- Horizontal + Vertical (rotation 180°)

### Visibilité

**Contrôle de la visibilité :**
- Case à cocher dans la liste des fragments
- Boutons "Afficher Tout" / "Masquer Tout"

**Utilité :**
- Comparaison de différentes configurations
- Isolation de fragments spécifiques
- Préparation pour l'exportation

## Sélection de Groupe

### Activation du Mode Sélection Rectangle

1. **Menu Édition → Outil de Sélection Rectangle** ou `Ctrl+Shift+R`
2. **L'outil est maintenant actif** (indiqué dans la barre de statut)

### Sélection de Plusieurs Fragments

1. **Cliquez et glissez** pour dessiner un rectangle de sélection
2. **Tous les fragments intersectant** le rectangle sont sélectionnés
3. **Le panneau de contrôle** bascule automatiquement sur l'onglet "Groupe"

### Manipulation de Groupe

**Rotation de Groupe :**
- **↺ 90° CCW** : Rotation antihoraire autour du centre du groupe
- **↻ 90° CW** : Rotation horaire autour du centre du groupe
- **Angle personnalisé** : Saisissez un angle libre pour le groupe
- **Rotation fine** : Ajustements par pas de 1°

**Déplacement de Groupe :**
- **Boutons fléchés** : Déplacement de tous les fragments sélectionnés
- **Glisser-déposer** : Cliquez sur un fragment du groupe et glissez

**Comportement :**
- Chaque fragment conserve sa rotation individuelle
- Le groupe tourne autour de son centre géométrique
- Les positions relatives sont préservées

### Désactivation

- **Décochez l'outil** dans le menu Édition
- **Cliquez sur un fragment seul** pour revenir à la sélection simple
- **Cliquez dans le vide** pour désélectionner tout

## Points Étiquetés

### Activation du Mode Points

1. **Menu Outils → Ajouter Point Étiqueté** ou `Ctrl+P`
2. **Le curseur devient une croix**
3. **La barre de statut** confirme l'activation

### Ajout de Points

1. **Cliquez sur un fragment** à l'endroit désiré
2. **Saisissez une étiquette** dans le dialogue (ex: "P1", "coin_gauche")
3. **Le point apparaît** avec son étiquette sur le fragment

### Étiquettes Correspondantes

**Principe :**
- Utilisez la **même étiquette** sur différents fragments
- Les points avec étiquettes identiques sont considérés comme correspondants
- Exemple : "P1" sur fragment A et "P1" sur fragment B

### Suture par Points

1. **Ajoutez des points correspondants** sur au moins 2 fragments
2. **Menu Outils → Suturer par Étiquettes** ou `Ctrl+Shift+S`
3. **L'algorithme aligne** les fragments basé sur les points correspondants

### Gestion des Points

**Nettoyage :**
- **Menu Outils → Effacer Tous les Points** : Supprime tous les points

**Visualisation :**
- Points rouges avec étiquettes blanches
- Visibles à tous les niveaux de zoom

## Suture Rigide

### Suture Automatique par Caractéristiques

**Principe :**
- Détection automatique de caractéristiques SIFT
- Correspondance entre fragments
- Optimisation des positions pour minimiser l'erreur

**Utilisation :**
1. **Positionnez grossièrement** les fragments manuellement
2. **Menu Outils → Suture Rigide** ou `Ctrl+S`
3. **L'algorithme affine** les positions automatiquement

### Suture par Points Étiquetés

**Principe :**
- Utilise les points étiquetés comme contraintes
- Plus précis que la suture automatique
- Contrôle total sur les correspondances

**Utilisation :**
1. **Ajoutez des points correspondants** (même étiquette)
2. **Menu Outils → Suturer par Étiquettes** ou `Ctrl+Shift+S`
3. **Les fragments s'alignent** sur les points correspondants

### Paramètres de Suture

**Suture Automatique :**
- Nombre de caractéristiques : 1000 par fragment
- Seuil de correspondance : 0.7
- Correspondances minimales : 10
- Seuil RANSAC : 5.0 pixels

**Optimisation :**
- Méthode : L-BFGS-B
- Itérations maximales : 1000
- Seuil de convergence : 1e-6

## Exportation

### Types d'Exportation

L'application propose plusieurs options d'exportation :

1. **Image Composite** : PNG rapide ou TIFF pyramidal
2. **Métadonnées** : Fichier JSON avec toutes les transformations

### Exportation d'Image

#### PNG (Aperçu Rapide)

**Caractéristiques :**
- Résolution unique
- Exportation rapide
- Idéal pour présentations

**Utilisation :**
1. **Cliquez sur "💾 Exporter"**
2. **Sélectionnez "PNG (Aperçu Rapide)"**
3. **Choisissez la qualité** (1-100%)
4. **Sélectionnez le fichier de sortie**

#### TIFF Pyramidal (Multi-Résolution)

**Caractéristiques :**
- Plusieurs niveaux de résolution
- Compatible avec OpenSlide et autres visualiseurs d'images pyramidales
- Fichier plus volumineux, temps d'exportation plus long

**Utilisation :**
1. **Cliquez sur "💾 Exporter"**
2. **Sélectionnez "TIFF Pyramidal (Multi-Résolution)"**
3. **Choisissez les niveaux** à exporter :
   - **Niveau 0** : Résolution complète
   - **Niveau 1** : 1/2 résolution
   - **Niveau 2** : 1/4 résolution
   - etc.
4. **Sélectionnez la compression** : LZW, JPEG, Deflate, ou Aucune
5. **Confirmez l'exportation**

### Sélection des Niveaux Pyramidaux

**Niveaux Communs :**
- Disponibles dans tous les fragments chargés
- Recommandés pour éviter les problèmes
- Affichés en **gras et bleu**

**Niveaux Partiels :**
- Disponibles seulement dans certains fragments
- Peuvent causer des problèmes d'alignement
- Affichés en gris avec avertissement

**Boutons de Sélection Rapide :**
- **Sélectionner Tout** : Tous les niveaux disponibles
- **Sélectionner Aucun** : Désélectionne tout
- **Sélectionner Communs** : Seulement les niveaux communs (recommandé)

### Exportation de Métadonnées

**Contenu du fichier JSON :**
```json
{
  "version": "1.0",
  "export_timestamp": "2024-01-15T10:30:00",
  "fragments": [
    {
      "id": "fragment-uuid",
      "name": "image1.tiff",
      "file_path": "/path/to/image1.tiff",
      "original_size": [2048, 1536],
      "transform": {
        "x": 100.5,
        "y": 200.3,
        "rotation": 90.0,
        "flip_horizontal": false,
        "flip_vertical": false
      },
      "display": {
        "visible": true,
        "opacity": 1.0
      }
    }
  ]
}
```

**Utilisation :**
1. **Menu Fichier → Exporter Métadonnées** ou `Ctrl+M`
2. **Choisissez l'emplacement** du fichier JSON
3. **Le fichier contient** toutes les transformations pour reproduction

## Raccourcis Clavier

### Fichier
- `Ctrl+O` : Charger des images
- `Ctrl+E` : Exporter l'image
- `Ctrl+M` : Exporter les métadonnées
- `Ctrl+Q` : Quitter l'application

### Édition
- `Ctrl+Shift+R` : Activer/désactiver la sélection rectangle
- `Ctrl+R` : Réinitialiser toutes les transformations
- `Suppr` : Supprimer le fragment sélectionné

### Affichage
- `Ctrl+0` : Zoom pour ajuster
- `Ctrl+1` : Zoom 100%

### Outils
- `Ctrl+S` : Suture rigide automatique
- `Ctrl+P` : Mode ajout de points étiquetés
- `Ctrl+Shift+S` : Suture par étiquettes

### Navigation
- **Molette souris** : Zoom in/out
- **Clic milieu + glisser** : Panoramique
- **Clic gauche + glisser** : Déplacer fragment

## Dépannage

### Problèmes de Démarrage

**Erreur : "No module named 'PyQt6'"**
```bash
pip install PyQt6==6.6.1
```

**Erreur : "OpenSlide library not found"**
- Windows : Installez OpenSlide depuis openslide.org
- macOS : `brew install openslide`
- Linux : `sudo apt-get install openslide-tools`

**Erreur : "OpenGL not supported"**
- Mettez à jour les pilotes graphiques
- Vérifiez le support OpenGL 2.1+

### Problèmes de Performance

**Chargement lent des images :**
- Utilisez des images de taille raisonnable (< 4 Go)
- Fermez les autres applications gourmandes en mémoire
- Augmentez la RAM disponible

**Interface qui rame :**
- Réduisez le nombre de fragments visibles
- Utilisez un zoom plus faible
- Fermez et relancez l'application

### Problèmes de Fonctionnalité

**Les boutons de groupe ne fonctionnent pas :**
1. Vérifiez que plusieurs fragments sont sélectionnés
2. Assurez-vous que l'outil rectangle est activé
3. Regardez les messages de débogage dans la console

**L'exportation échoue :**
- Vérifiez l'espace disque disponible
- Assurez-vous d'avoir les permissions d'écriture
- Essayez un format différent (PNG au lieu de TIFF)

**La suture ne fonctionne pas :**
- Vérifiez que les fragments se chevauchent
- Ajoutez des points étiquetés manuellement
- Positionnez grossièrement les fragments avant la suture

### Messages d'Erreur Courants

**"No visible fragments to export"**
- Assurez-vous qu'au moins un fragment est visible (case cochée)

**"No pyramid levels selected"**
- Sélectionnez au moins un niveau dans le dialogue d'exportation TIFF

**"Feature matching failed"**
- Les fragments n'ont pas assez de caractéristiques communes
- Essayez la suture par points étiquetés

### Support et Débogage

**Mode débogage :**
```bash
python main.py --debug
```

**Logs détaillés :**
- Les messages apparaissent dans la console
- Recherchez les erreurs en rouge
- Les avertissements en jaune sont généralement non-critiques

**Informations système :**
```python
import sys, PyQt6, cv2, numpy
print(f"Python: {sys.version}")
print(f"PyQt6: {PyQt6.QtCore.qVersion()}")
print(f"OpenCV: {cv2.__version__}")
print(f"NumPy: {numpy.__version__}")
```

### Limitations Connues

1. **Taille maximale des images** : 8 Go par fragment
2. **Nombre de fragments** : Recommandé < 50 pour de bonnes performances
3. **Formats supportés** : Limité aux formats listés ci-dessus
4. **Images non-prétraitées** : Peut causer des problèmes d'alignement
5. **Suture** : Fonctionne mieux avec des fragments qui se chevauchent
6. **Mémoire** : Les images très haute résolution peuvent saturer la RAM
7. **Transparence** : Nécessite des images avec canal alpha correctement défini

---

## Contact et Support

Pour des questions techniques ou des rapports de bugs, consultez la documentation du projet ou créez une issue dans le dépôt.

**Version de ce README :** 1.0  
**Dernière mise à jour :** Janvier 2024