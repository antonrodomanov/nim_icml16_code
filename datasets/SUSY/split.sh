#!/bin/bash

echo "Splitting..."
split --verbose -l 4000000 SUSY SUSY_

echo "Done splitting. Rename the files."
mv SUSY_aa SUSY
mv SUSY_ab SUSY.t
