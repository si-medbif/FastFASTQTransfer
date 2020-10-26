for FILE in Data/Quality/*.quality
do
    FULL_FILE_NAME="${FILE##*/}"
    FILE_NAME="${FULL_FILE_NAME%.*}"
    echo $FILE_NAME
    
    for i in {1..10}
    do
        #Arithmetic Compression
        /usr/bin/time -f "%E,%U,%K,%M\n" cat $FILE | ./Experiments/Pilot_Compression/cacm-ac/adaptive_encode > Data/Temp_Arithmetic/$FILE_NAME.arithmetic

        # Arithmetic Decompression
        /usr/bin/time -f "%E,%U,%K,%M\n" cat Data/Temp_Arithmetic/$FILE_NAME.arithmetic | ./Experiments/Pilot_Compression/cacm-ac/adaptive_decode > Data/Temp_Arithmetic/$FILE_NAME
    done
done
