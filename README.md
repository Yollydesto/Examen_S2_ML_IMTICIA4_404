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

*(Rédigez ici 2 à 3 phrases expliquant le problème d’annulation rencontré par Atlantic Haven Hotels, ses conséquences opérationnelles et l’intérêt d’une prédiction suffisamment précoce.)*

#### Méthodologie adoptée

*(Résumez votre démarche : EDA, traitement des données, feature engineering, validation temporelle, baseline, modèles comparés et choix du seuil de décision.)*

#### Résultats obtenus

*(Indiquez le meilleur F1-score obtenu sur votre jeu de validation, les principales métriques complémentaires et une découverte importante issue de votre analyse.)*

#### Mots-clés

*(Indiquez cinq à huit mots-clés techniques ou métier, par exemple : classification binaire, annulation, validation temporelle, F1-score, feature engineering.)*

---

### **3. Contenu du Repository**

Voici la liste des fichiers et liens importants permettant d’évaluer votre travail :

- **notebook.ipynb** : code complet de l’EDA, du prétraitement, de la modélisation et de l’évaluation ;
- **submission.csv** : prédictions sur `reservations_test.csv` ;
- **README.md** : présent rapport complété ;
- **requirements.txt** : dépendances nécessaires à la reproduction du projet *(si nécessaire)* ;
- *(ajoutez ici les autres fichiers utiles sans inclure les fichiers temporaires).* 

**🔗 Liens utiles :**

- [**LIEN VERS LA VIDÉO DE PRÉSENTATION** — Google Drive ou YouTube](https://drive.google.com/drive/u/0/folders/1ReWGkGBJZxrbx6PlMtgF6_-l9iGfQo7x)
- [Lien vers le dépôt GitHub](https://github.com/Yollydesto/Examen_S2_ML_IMTICIA4_404)
- [Lien vers une autre ressource — facultatif](https://www.google.com/)

---

### **4. Résultats de Modélisation**

Présentez les résultats obtenus sur **le même jeu de validation** afin que la comparaison soit valide.

| Modèle | Paramètres principaux | F1-score | Précision | Rappel | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Régression logistique — baseline |  |  |  |  |  |
| Modèle 2 |  |  |  |  |  |
| Modèle 3 |  |  |  |  |  |
| Modèle final |  |  |  |  |  |

**Seuil de décision retenu :** *(votre réponse ici)*

**Justification du choix du modèle final :**

*(Votre réponse ici. Ne vous limitez pas au score : considérez la stabilité, l’interprétabilité, les erreurs et le coût métier.)*

---

### **5. Réponses aux Questions d’Analyse**

*Répondez précisément aux questions ci-dessous. Utilisez des chiffres, tableaux ou références à vos graphiques pour justifier vos réponses.*

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

*(Présentez au moins une comparaison chiffrée et discutez les limites liées aux petits sous-groupes.)*

#### **Q9. Analyse des erreurs**

Analysez au minimum :

- cinq faux positifs ;
- cinq faux négatifs ;
- les raisons possibles de ces erreurs ;
- une piste d’amélioration des données ou du modèle.

*(Votre réponse ici.)*

---

### **6. Conclusion et Recommandations**

*(Résumez en un court paragraphe les performances, les limites et les conditions raisonnables d’utilisation du modèle.)*

**Recommandation opérationnelle finale :**

*(Votre réponse ici.)*

---

### **7. Reproductibilité**

- version de Python :
- principales bibliothèques et versions :
- graine(s) aléatoire(s) :
- commande ou procédure d’exécution :
- durée approximative d’entraînement :
- environnement utilisé : *(local, Google Colab, Kaggle, etc.)*

---

### **8. Bibliographie**

*(Listez les livres, articles, documentations et liens ayant servi dans ce travail. Mentionnez également les outils d’IA générative utilisés et décrivez brièvement leur contribution.)*

- Référence 1 :
- Référence 2 :
- Référence 3 :
