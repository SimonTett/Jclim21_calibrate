#!/bin/bash
export OPTCLIMTOP="/exports/csce/datastore/geos/groups/OPTCLIM/grl17_coupled/software/optimise2017"

SCRIPT=$OPTCLIMTOP/tools/comp_obs_netcdf.py 

INPUTDIR=/exports/csce/datastore/geos/groups/China-Workshop/000DataAMIP-Simon/
for i in {1..90} # have 90 models
 do 
    Pattern="Model"$i"GRID_200103-200502_"
    files=$(ls "$INPUTDIR/"*"/$Pattern"*".nc")
    outputFile="Model"$i"_obs.nc"
    $SCRIPT $OPTCLIMTOP/Configurations/example_ds.json $outputFile $files
    echo "$Pattern -> $outputFile"
done
