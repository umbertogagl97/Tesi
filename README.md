# Tesi
## Addestramento_modello
É usato per creare un nuovo modello addestrandolo con immagini caricate tramite funzioni, quindi non è utile per immagini presenti in cartelle su drive(per ora utilizza solo tensorflow e keras) (è da eliminare).
## Attacco_base
Importa un modello da drive e lo attacca usando ART permettendo di stampare un subplot del risultato, fare testing e di attaccare singole immagini prelevate da drive.
## Addestramento modello pytorch
Notebook che addestra un modello pytorch e lo attacca con FGM (è da eliminare)
## Richieste prof
Cartella con notebook richiesti dal prof, finora è stato realizzata solo una variante di FGM che calcola perturbazioni in scala di grigi
## Impronte
Cartella con notebooks che utilizzano il dataset con le impronte
### Addestramento Scanner
#### Creazione dataset
Crea il dataset su Drive costituito dalle patch delle immagini originali (data augmentation offline)
#### Addestramento scanner pytorch (test media)
Addestra un singolo scanner, lo salva su drive e testa la sua accuracy sul test set originale andando a fare la media delle predizioni delle 10 patch di ogni immagine
### AttaccoScanner 
#### AttaccoScanner pytorch
Importa modello da Drive, lo attacca e mostra i risultati di testing prima e dopo l'attacco
#### Attacco FGM final
Variante di FGM che, settando minimal=True, applica le perturbazioni solo alle immagini spoof predette bene in maniera da far predire tutte le immaigni spoof come live.
Final_noresize è un'ulteriore variante che attacca un'immagine alla volta ed effettua contemporaneamente il test sull'img_adv senza effettuare il resize (224,224), in questo modo, a causa della saturazione della RAM, non si riescono a memorizzare le img_adv ma si effettua un test identico a quello effettuando durante l'addestramento, senza alterazioni dovute al resize.
