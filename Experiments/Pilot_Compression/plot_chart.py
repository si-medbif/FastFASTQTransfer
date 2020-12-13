import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Pilot the result from result_extractor.py
# INPUT: <PLOT PATH> <SAMPLE SIZE FILE> <RESULT FILE>

def plot_compression_time (df, dest_path, palette) :
    plt.figure(figsize=(14,8))
    sns.barplot(data=df, x='sample_name', y='c_real_time', hue='Method', palette=palette)
    
    plt.title('Compression time in each method and sample')
    plt.xlabel('Sample')
    plt.ylabel('Time (sec)')
    plt.savefig(dest_path, dpi=300)
    plt.clf()

def plot_decompression_time (df, dest_path, palette) :
    plt.figure(figsize=(14,8))
    sns.barplot(data=df, x='sample_name', y='d_real_time', hue='Method', palette=palette)
    
    plt.title('Decompression time in each method and sample')
    plt.xlabel('Sample')
    plt.ylabel('Time (sec)')
    plt.savefig(dest_path, dpi=300)
    plt.clf()

def plot_compressed_size (df, dest_path, palette) :
    df['compressed_size'] = df['compressed_size'] / (10**9)
    plt.figure(figsize=(14,8))
    sns.barplot(data=df, x='sample_name', y='compressed_size', hue='Method', palette=palette)
    
    plt.title('Compressed size from each method and sample')
    plt.xlabel('Sample')
    plt.ylabel('Size (GB)')
    plt.savefig(dest_path, dpi=300)
    plt.clf()

def plot_compression_max_memory (df, dest_path, palette) :
    df['c_max_memory'] = df['c_max_memory'] / 1000
    plt.figure(figsize=(14,8))
    ax = sns.barplot(data=df, x='sample_name', y='c_max_memory', hue='Method', palette=palette)
    ax.set_yscale("log")
    
    plt.title('Peak compression memory from each method and sample')
    plt.xlabel('Sample')
    plt.ylabel('Memory (MB)')
    plt.savefig(dest_path, dpi=300)
    plt.clf()

def plot_decompression_max_memory (df, dest_path, palette) :
    df['d_max_memory'] = df['d_max_memory'] / 1000
    plt.figure(figsize=(14,8))
    ax = sns.barplot(data=df, x='sample_name', y='d_max_memory', hue='Method', palette=palette)
    ax.set_yscale("log")
    
    plt.title('Peak decompression memory from each method and sample')
    plt.xlabel('Sample')
    plt.ylabel('Memory (MB)')
    plt.savefig(dest_path, dpi=300)
    plt.clf()

def plot_compression_rate (df, dest_path, palette) :
    df['compression_rate'] = df['original_size'] / df['c_real_time']

    plt.figure(figsize=(14,8))
    sns.barplot(data=df, x='sample_name', y='compression_rate', hue='Method', palette=palette)
    
    plt.title('Compression rate in each method and sample')
    plt.xlabel('Sample')
    plt.ylabel('Compression Rate (MB/s)')
    plt.savefig(dest_path, dpi=300)
    plt.clf()

def plot_decompression_rate (df, dest_path, palette) :
    df['decompression_rate'] = df['original_size'] / df['d_real_time']

    plt.figure(figsize=(14,8))
    sns.barplot(data=df, x='sample_name', y='decompression_rate', hue='Method', palette=palette)
    
    plt.title('Decompression rate in each method and sample')
    plt.xlabel('Sample')
    plt.ylabel('Decompression Rate (MB/s)')
    plt.savefig(dest_path, dpi=300)
    plt.clf()

def plot_decreased_percentage (df, dest_path, palette) :  
    df['decreased_rate'] = ((df['original_size'] - df['compressed_size']) / df['original_size']) * 100
    
    plt.figure(figsize=(14,8))
    sns.barplot(data=df, x='sample_name', y='decreased_rate', hue='Method', palette=palette)
    
    plt.title('Size decreased percentage for each sample and method')
    plt.xlabel('Sample')
    plt.ylabel('Decreased Size Percentage')
    plt.savefig(dest_path, dpi=300)
    plt.clf()

def plot_avg_decreased_percentage (df, dest_path) :
    df['decreased_rate'] = ((df['original_size'] - df['compressed_size']) / df['original_size']) * 100

    plt.figure(figsize=(14,8))
    sns.barplot(data=df, x='Method', y='decreased_rate')
    
    plt.title('Size decreased percentage for each method')
    plt.xlabel('Method')
    plt.savefig(dest_path, dpi=300)
    plt.clf()

def main (args) :
    plot_path = args[1]
    sample_size_path = args[2]
    result_file_list = args[3:]
    whole_df = pd.DataFrame()
    original_file_size = pd.read_csv(sample_size_path)

    for result_file in result_file_list :
        sub_dataset = pd.read_csv(result_file)
        whole_df = pd.concat([whole_df, sub_dataset])
    
    whole_df = pd.merge(whole_df, original_file_size, on=['sample_name'])
    whole_df = whole_df.sort_values(by=['original_size', 'sw_name'])

    whole_df = whole_df.rename(columns={'sw_name' : 'Method'})

    colour_unique = whole_df['Method'].unique()
    palette = dict(zip(colour_unique, sns.color_palette(n_colors=len(colour_unique))))

    plot_compression_time(whole_df, plot_path + '/compression_time.png', palette)
    plot_decompression_time(whole_df, plot_path + '/decompression_time.png', palette)
    plot_compressed_size (whole_df, plot_path + '/compressed_size.png', palette)
    plot_compression_max_memory (whole_df, plot_path + '/compression_max_memory.png', palette)
    plot_decompression_max_memory (whole_df, plot_path + '/decompression_max_memory.png', palette)
    plot_compression_rate(whole_df, plot_path + '/compression_rate.png', palette) 
    plot_decompression_rate(whole_df, plot_path + '/decompression_rate.png', palette)
    plot_decreased_percentage(whole_df, plot_path + '/decreased_rate.png', palette)
    plot_avg_decreased_percentage(whole_df, plot_path + '/whole_decreased_rate.png')

if __name__ == "__main__":
    main(sys.argv)