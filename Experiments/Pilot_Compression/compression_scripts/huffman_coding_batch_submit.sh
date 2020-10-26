for FILE in Data/Quality/*.quality
do
    FULL_FILE_NAME="${FILE##*/}"
    FILE_NAME="${FULL_FILE_NAME%.*}"
    echo $FILE_NAME
    
    for i in {1..10}
    do
        #Huffman Compression
        /usr/bin/time -f "%E,%U,%K,%M\n" python Experiments/Pilot_Compression/Huffman-Coding/Huffman.py -c $FILE

        # Move to Temp Folder
        mv Data/Quality/$FILE_NAME.huffman Data/Temp_Huffman

        # Huffman Decompression
        /usr/bin/time -f "%E,%U,%K,%M\n" python Experiments/Pilot_Compression/Huffman-Coding/Huffman.py -d  Data/Temp_Huffman/$FILE_NAME.huffman

        # Remove Temp
        rm Data/Temp_Huffman/$FILE_NAME.huffman
        rm Data/Temp_Huffman/$FILE_NAME.quality
    done
done
