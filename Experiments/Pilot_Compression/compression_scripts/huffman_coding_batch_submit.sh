#$ -S /bin/sh
#$ -cwd
#$ -j y

export PATH=/colossus/home/arnon:$PATH
source /share/apps/anaconda2/bin/activate FFQT

for FILE in Data/Quality_Small_Batch/*.quality
do
    FULL_FILE_NAME="${FILE##*/}"
    FILE_NAME="${FULL_FILE_NAME%.*}"
    echo $FILE_NAME
    
    for i in {1..10}
    do
        #Huffman Compression
        /usr/bin/time -f "%e,%S,%U,%P,%K,%M" python Experiments/Pilot_Compression/Huffman-Coding/Huffman.py -c $FILE

        # Move to Temp Folder
        cp Data/Quality_Small_Batch/$FILE_NAME.huffman Data/Temp_Huffman

	# Measure Compress File Size
	stat -c %s Data/Temp_Huffman/$FILE_NAME.huffman

        # Huffman Decompression
        /usr/bin/time -f "%e,%S,%U,%P,%K,%M" python Experiments/Pilot_Compression/Huffman-Coding/Huffman.py -d  Data/Temp_Huffman/$FILE_NAME.huffman

        # Remove Compressed File
        rm Data/Temp_Huffman/$FILE_NAME.huffman
    done
done
