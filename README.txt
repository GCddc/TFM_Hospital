README
* 1 primero usar el  Notebook Unclassified_uncultured. Se generar  un archivo llamado: Diversidad Hospital_UN_trated_Hpy.xlsx

* 2 segundo usar el Notebook Columscreator. Se generar  un archivo llamado: Diversidad Hospital_columscreator_Hpy.xlsx


* 3 tercero usar el Notebook Merge. Se generar  el archivo Excel definitivo:      Diversidad Hospital_merge_Hpy.xlsx

* 4  Ya puedes usar el notebook de visualización donde se generan la mayor parte de las gráficas.
* 5 El gráfico KRONA: 
Los documentos que se generan con el Notebook Krona, que se guardar n a la carpeta Krona

Posteriormente una vez generados, se necesita un programa que aparece en https://github.com/marbl/Krona/wiki 
      Se instala según estas instrucciones: https://github.com/marbl/Krona/wiki/Installing
se ejecuta en Linux con el comando ktImport *.txt desde la carpeta Krona que recoge todos los txt que encuentra en la carpeta krona y genera el gráfico. Dejo en esta carpeta los que he generado con el Notebook Krona. Y el gráfico generado es un archivo HTML que se ve en cualquier explorador(Krona_Hospital.html).

* 6 EL RDA se ejecuta en R studio, el entorno y los exceles que utiliza están en la misma carpeta y los gráficos que he generado, están también ahi guardados como archivos jpg.

* 7 El Frontend se carga desde el términal, accediendo primero a la carpeta donde se encuentra, y luego usando el comando 'streamlit run frontend.py'.
