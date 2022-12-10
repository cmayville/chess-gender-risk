#!/bin/sh

gamesfolder="selected_games"

mkdir selected_games/20

cd $gamesfolder
for FILE in *
do
    echo $FILE
    lines=$(wc -l $FILE | awk '{print $1}')
    if [ "21" -lt "$lines" ]
    then
        mv $FILE ../selected_games/20/$FILE
    fi
done
