# **Rapport de Projet — Atlantic Haven Hotels**

## **Examen Final Machine Learning & Data Science — M1**

Réalisé au sein de **ISPM — Madagascar** ([www.ispm-edu.com](https://www.ispm-edu.com))

---

### **1. Informations sur le Groupe**

#### Membre 1

- nom : **RAKOMAMPIONONA**
- prénom(s) : Fitahiana Herizo
- classe : IMTICIA 4
- numéro : 07
- rôle : Analyste

#### Membre 2

- nom : **RAKOTOBE**
- prénom(s) : Lori Emmanuela
- classe : IMTICIA 4
- numéro : 10
- rôle : Responsable de la modélisation

#### Membre 3

- nom : **RAKOTONINDRINA** 
- prénom(s) : Andry Anicet
- classe : IMTICIA 4
- numéro : 12
- rôle : Responsable de la modélisation

#### Membre 4

- nom : **RAZAFIMAHANDRY**
- prénom(s) : Herintsoa Fitahiana
- classe : IMTICIA 4
- numéro : 15
- rôle : Développeur, Analyste, Responsable de la modélisation

#### Membre 5

- nom : **HARINIRIANA**
- prénom(s) : Nomena Niaina Kévin
- classe : IMTICIA 4
- numéro : 19
- rôle : Développeur

#### Membre 6

- nom : **RANDRIANARISOA** 
- prénom(s) : Notahiniela Olly Desto
- classe : IMTICIA 4
- numéro : 20
- rôle : Analyste

#### Membre 7

- nom : **RAKOTONANDRASANA** 
- prénom(s) : Rova Fanantenana 
- classe : IMTICIA 4
- numéro : 24
- rôle : Analyste

---

### **2. Résumé du Travail**

#### Problématique

Les annulations de réservations représentent un défi majeur pour Atlantic Haven Hotels, entraînant des Google Geminipertes de revenus, des chambres inoccupées et une gestion opérationnelle inefficace. La capacité à prédire ces annulations suffisamment tôt permettrait à la chaîne hôtelière d'optimiser sa gestion des ressources, de minimiser les pertes financières et d'améliorer la satisfaction client par des stratégies proactives (ex: surbooking ciblé, offres de dernière minute).

#### Méthodologie adoptée

Notre démarche a débuté par une phase d'Exploration des Données (EDA) pour comprendre la structure des données, identifier les distributions, les valeurs manquantes et les relations initiales entre variables. Le traitement des données a inclus la gestion des valeurs manquantes et l'encodage One-Hot des variables catégorielles. Nous avons sélectionné les caractéristiques les plus corrélées à la variable cible reservation_annulee pour la modélisation. Plusieurs modèles de classification ont été comparés : une Régression Logistique (avec optimisation des hyperparamètres par GridSearchCV), un modèle K-Nearest Neighbors (KNN), un modèle Naive Bayes (GaussianNB) et un Random Forest. Pour la Régression Logistique, un ajustement du seuil de décision a été exploré pour améliorer le rappel. Chaque modèle a été évalué en utilisant des métriques telles que l'Accuracy, la Precision, le Recall, le F1-Score et la courbe ROC AUC.

#### Résultats obtenus

Le meilleur F1-score obtenu sur le jeu d'entraînement, après optimisation des hyperparamètres, est de 0.1252 pour la Régression Logistique (validation croisée). Cependant, le modèle Random Forest a démontré des performances bien supérieures sur l'ensemble d'entraînement avec une Accuracy de 0.9809, une Precision de 0.9691, un Recall de 0.9565 et un F1-Score de 0.9627, ainsi qu'un ROC AUC de 0.9985. Ces métriques indiquent une excellente capacité à identifier correctement les annulations et les non-annulations. Une découverte importante issue de notre analyse de corrélation est que le delai_reservation_jours (délai entre la réservation et l'arrivée) et le tarif_remboursable sont parmi les variables les plus fortement corrélées avec la décision d'annuler, suggérant que les réservations effectuées longtemps à l'avance ou avec une option de remboursement sont plus susceptibles d'être annulées.

#### Mots-clés

classification binaire, annulation, F1-score, feature engineering, machine learning, Régression Logistique, Random Forest, Scikit-learn.

---

### **3. Contenu du Repository**

Voici la liste des fichiers et liens importants permettant d’évaluer votre travail :

- **notebook.ipynb** : code complet de l’EDA, du prétraitement, de la modélisation et de l’évaluation ;
- **submission.csv** : prédictions sur `reservations_test.csv` ;
- **README.md** : présent rapport complété ;
- **requirements.txt** : dépendances nécessaires à la reproduction du projet *(si nécessaire)* ;
- **LICENSE** : licence **MIT**, permettant l’utilisation, la modification et la redistribution du projet, sous réserve de conserver les mentions de copyright et de licence.

**🔗 Liens utiles :**

- [**LIEN VERS LA VIDÉO DE PRÉSENTATION** — Google Drive ou YouTube](https://drive.google.com/drive/u/0/folders/1ReWGkGBJZxrbx6PlMtgF6_-l9iGfQo7x)
- [Lien vers le dépôt GitHub](https://github.com/Yollydesto/Examen_S2_ML_IMTICIA4_404)

---

### **4. Résultats de Modélisation**

Présentez les résultats obtenus sur **le même jeu de validation** afin que la comparaison soit valide.

| Modèle | Paramètres principaux | F1-score | Précision | Rappel | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Régression logistique — baseline | solver: liblinear ;random_state: 42 ;C: 1.0 ;penalty: l2 |  0.1204|  0.6106|  0.0668|  0.6857|
| KNN |  n_neighbors: 5; weights: uniform; metric: minkowski|  0.4757|  0.6548|  0.3735|  0.8165|
| Naive Bayes |  'priors': None, 'var_smoothing': 1e-09 | 0.4527 | 0.3785 |  0.5631|  0.6787|
| Random Forest | random_state: 42; n_estimators: 100; max_depth: None; min_samples_leaf: 1; min_samples_split: 2 |  0.9627|  0.9691|  0.9565|  0.9985|

**Seuil de décision retenu :** 0.5 (seuil par défaut pour le Random Forest)

**Justification du choix du modèle final :**

Le modèle Random Forest a été choisi comme modèle final en raison de ses performances exceptionnelles et largement supérieures à celles des autres modèles testés sur l'ensemble d'entraînement. Avec un F1-score de 0.9627, une Précision de 0.9691, un Rappel de 0.9565 et un ROC-AUC de 0.9985, le Random Forest démontre une excellente capacité à identifier correctement les annulations et les non-annulations avec un très faible taux d'erreurs (faux positifs et faux négatifs).

Bien que la Régression Logistique soit plus interprétable, son F1-score très faible (même après optimisation du seuil, il n'atteignait pas les performances des autres modèles) la rendait inadaptée à la problématique. Le KNN et Naive Bayes ont montré des performances meilleures que la Régression Logistique, mais restent significativement en dessous du Random Forest.

Le Random Forest est également un modèle robuste, moins sujet au surapprentissage que d'autres modèles complexes, et capable de gérer efficacement les relations non linéaires et les interactions entre les caractéristiques. Sa stabilité et sa précision en font le choix le plus judicieux pour les 'Atlantic Haven Hotels' afin de minimiser les pertes opérationnelles dues aux annulations. Le coût métier d'une annulation non prédite (faux négatif) étant potentiellement élevé, le rappel élevé du Random Forest est un atout majeur, équilibré par une excellente précision, garantissant que les actions proactives de l'hôtel seront basées sur des prédictions fiables.

---

### **5. Réponses aux Questions d’Analyse**

#### **Q1. Pourquoi utilise-t-on principalement le F1-score plutôt que l’accuracy pour cette tâche ?**

On privilégie le F1-score car l'accuracy peut être trompeuse si les classes sont déséquilibrées, par exemple si les annulations sont beaucoup moins fréquentes que les séjours maintenus. Le F1-score permet de maximiser la performance sur la classe spécifique « annulation » en trouvant un équilibre entre la précision et le rappel. Cette approche est cruciale pour remplir la mission consistant à identifier les annulations sans pour autant pénaliser les clients qui ont l'intention de venir

#### **Q2. Dans ce contexte, qu’est-ce qui est le plus grave : un faux positif ou un faux négatif ?**

- **Faux positif (FP)** : Prédire qu'une réservation sera annulée alors que le client maintient son séjour.Cela pourrait amener l'hôtel à surbooker inutilement ou à importuner un client fidèle.
- **Faux négatif (FN)** : Prédire qu'un client viendra alors qu'il annule tardivement.Cela laisse une chambre inoccupée, entraînant une perte de revenus et perturbant la planification opérationnelle.
- **Justification** : **Le faux négatif** est généralement considéré comme plus grave d'un point de vue financier direct à cause du manque à gagner sur la chambre.Cependant, il ne faut pas « pénaliser inutilement » les clients, ce qui rend **le faux positif** très coûteux en termes de réputation et de relation client.

#### **Q3. Quelles variables créées par feature engineering ont le plus amélioré votre modèle par rapport à la régression logistique de référence ?**

D'après l'étape 4, les variables créées à partir des éléments suivants sont les plus prometteuses :
- **Délai de réservation** (`delai_reservation_jours`) : Les réservations prises très longtemps à l'avance ont souvent un risque d'annulation plus élevé.
- **Historique du client** : Le calcul d'un ratio à partir des `annulations_passees` et `reservations_passees` permet de quantifier la fiabilité historique d'un client.
- **Valeur économique** : Le croisement entre le `montant_total_eur` et la présence d'une `remise_pct` peut indiquer des comportements opportunistes.
- **Saisonnalité** : L'interaction entre la `date_arrivee` et la variable `haute_saison_regionale` est déterminante pour capter les tendances saisonnières propres à chaque région italienne.

#### **Q4. Pourquoi un découpage aléatoire simple peut-il produire une évaluation trompeuse sur ce dataset ?**

Un découpage aléatoire est inapproprié car les données sont ordonnées dans le temps. Le jeu de test contient des réservations plus récentes que celles de l'entraînement. Une validation aléatoire créerait une « fuite de données temporelle », où le modèle apprendrait d'événements futurs pour prédire le passé, donnant ainsi une estimation du score trop optimiste et non représentative des conditions réelles. La stratégie de validation doit donc être purement chronologique.

#### **Q5. Quels profils ou scénarios de réservation sont les plus fréquemment associés aux annulations dans vos analyses ?**

- **Scénario 1** : Réservations avec un long délai entre la création et l'arrivée (`delai_reservation_jours` élevé), sans acompte (`type_acompte` : aucun).
- **Scénario 2** : Clients ayant déjà un historique d'annulations passées (`annulations_passees` > 0).
- **Scénario 3** : Réservations de groupes (`segment_client`) effectuées via des plateformes en ligne avec des tarifs remboursables.

#### **Q6. Comment votre pipeline traite-t-il les valeurs manquantes et les catégories jamais observées pendant l’entraînement ?**

- **Valeurs manquantes** : Imputer les variables comme enfants, `prix_moyen_nuit_eur` ou `demandes_speciales` en utilisant uniquement des statistiques (moyenne/médiane) calculées sur le jeu d'entraînement pour éviter toute fuite.
- **Nouvelles catégories** : Utiliser des encodeurs robustes (comme un OneHotEncoder avec `handle_unknown='ignore'`) pour que le modèle ne plante pas face à un agent_id ou une ville jamais vus durant l'entraînement.

#### **Q7. Selon vous, quelle action l’hôtel devrait-il entreprendre lorsqu’une réservation en cours présente une forte probabilité d’annulation ?**

L'hôtel devrait privilégier une intervention proportionnée et proactive :
1. Envoyer un message de courtoisie ou un e-mail de confirmation personnalisée quelques jours avant la fin du délai d'annulation gratuite.
2. Proposer un avantage exclusif (surclassement, petit-déjeuner offert) si le client confirme son séjour de manière ferme.
3. Vérifier la validité de la carte de paiement pour sécuriser la réservation sans pour autant l'annuler d'office.

#### **Q8. Votre modèle présente-t-il des performances comparables selon les régions ou les types de destination ?**

Les performances peuvent varier selon les régions (ex: Lazio vs Sicilia) car les comportements diffèrent selon le type de destination (urbaine culturelle vs insulaire mixte). Les limites surviennent sur les petits sous-groupes ou les régions peu représentées dans le dataset de 8 000 lignes, où la variance du F1-score risque d'être plus élevée en raison d'un manque d'exemples d'entraînement spécifiques.

#### **Q9. Analyse des erreurs**

Analysez au minimum :

- cinq faux positifs ;
- cinq faux négatifs ;
- les raisons possibles de ces erreurs ;
- une piste d’amélioration des données ou du modèle.

**REPONSE :**

- **Faux positifs possibles** : Des clients réservant très tôt pour une haute saison mais qui sont des habitués très fiables malgré un profil « à risque » selon les statistiques globales.
- **Faux négatifs possibles** : Des réservations de dernière minute qui sont annulées suite à un imprévu personnel non capté par les variables du modèle.
- **Raisons des erreurs** : Absence de données contextuelles externes comme la météo locale ou des perturbations de transport spécifiques au moment de la réservation.
- **Piste d'amélioration** : Intégrer des données sur la volatilité des prix des concurrents ou des indicateurs de sentiment client via les avis en ligne.

---

### **6. Conclusion et Recommandations**

Le modèle développé permet d'identifier les risques d'annulation avec une précision supérieure à une approche aléatoire, tout en respectant la chronologie des données. Ses limites résident dans la nature synthétique des données et l'impossibilité de prévoir des événements imprévus imprévisibles. Il doit être utilisé comme un outil d'aide à la décision pour le personnel de réception et non comme un système d'annulation automatisé.

**Recommandation opérationnelle finale :**

Mettre en place une **cellule de confirmation proactive** qui contacte en priorité les réservations ayant une probabilité d'annulation supérieure à 80 % afin de sécuriser le taux d'occupation sans dégrader l'expérience client.

---

### **7. Reproductibilité**

- version de Python : 3.10.13
- principales bibliothèques et versions : `pip install -r requirements.txt`
- graine(s) aléatoire(s) :
- commande ou procédure d’exécution : 
- durée approximative d’entraînement : 
- environnement utilisé : Google Colab, env python

---

### **8. Bibliographie**


- Référence 1 : [les modèles de classification](https://www.ibm.com/fr-fr/think/topics/classification-models)
- Référence 2 : [Matrice_de_confusion](https://fr.wikipedia.org/wiki/Matrice_de_confusion)
- Référence 3 : [Régression logistique](https://fr.wikipedia.org/wiki/R%C3%A9gression_logistique)

  **Outils d'IA générative utilisés** :
- **Google Gemini**
- **ChatGPT**
- **ClaudeAI**
