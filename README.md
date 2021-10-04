# Tesi
## Addestramento_modello
É usato per creare un nuovo modello addestrandolo con immagini caricate tramite funzioni, quindi non è utile per immagini presenti in cartelle su drive(per ora utilizza solo tensorflow, prossima implementazione userà anche pytorch)
## Attacco_base
Importa un modello da drive e lo attacca usando ART, permettendo di stampare un subplot del risultato, fare testing e permette anche di attaccare singole immagini prelevate da drive (per ora contiene solo l'attacco FGM, ma in seguito sarà creato un file simile con vari attacchi a confronto)
## Richieste prof
Cartella con notebook che richiede il prof, finora è stato realizzata solo una variante di FGM che calcola perturbazioni in scala di grigi
## Impronte
Cartella con notebook che utilizzano il dataset con le impronte
### Addestramento modello
Sono presenti due notebook che creano i modelli (keras o pytorch) e li salvano su Drive
### Attacco
TODO: notebook che importa modello da Drive e lo attacca
