# county-field-clean
A cleaning script to correct county fields in natural history records

This simple python script requires a Darwin Core archive zipfile and references the .csv file provided in ../data/

Many of the incorrectly transcribed county fields are actually lower level administrative levels. Roughly 40% of Tennessee's county errors ~~fall~~ fell into this category. Referencing census data, a county can often be safely inferred using state, and lower administrative levels. This method also allows salvaging the municipality field.

This script also, explicitly addresses many errors found to be common in Tennessee collections.

Back up, and verify any/all results for yourself before accepting the changes. I highly reccomend initially using the -r flag when testing this script.

### Usage

Command list available with the -h command.
Ex: countyCleaner.py -h

It is a wise idea to first see the reccomendations the script will make, using the -r flag creates a simplified report with the suggested changes.

Ex: countyCleaner.py -f pathToDarwinCoreArchive.zip -r

To actually clean records simply use:

Ex: countyCleaner.py -f pathToDarwinCoreArchive.zip
