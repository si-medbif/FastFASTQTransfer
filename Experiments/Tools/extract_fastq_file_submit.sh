#$ -S /bin/sh
#$ -cwd
#$ -j y

export PATH=/colossus/home/arnon/1000Genomes:$PATH

for FILE in Data/Samples/*.fastq
do
    FULL_FILE_NAME="${FILE##*/}"
    FILE_NAME="${FULL_FILE_NAME%.*}"
    echo $FILE_NAME
    
    # Extracting Quality
    awk '0 == (NR + 4) % 4' $FILE > Data/Quality/$FILE_NAME.quality

    # Extracting Read
    awk '0 == (NR + 2) % 4' $FILE > Data/Read/$FILE_NAME.read
done

