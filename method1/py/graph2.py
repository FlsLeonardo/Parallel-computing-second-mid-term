import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

def main():
        #for num in range(4, 13):
            #efficency_speedup("../output/Omp.csv",num)
        #speedup("../output/Omp.csv")
        #efficiency("../output/Omp.csv")
        efficiencyOMP_MPI("../output/Omp.csv","../output/MPI.csv","../output/Serial.csv")
  
def efficency_speedup(filename,dim_matrix):
    matrix_times = []
    matrix_thread_used = []
    speedup = []
    efficency = []
    tempo_seriale = 0
    with open(filename, mode='r', encoding='utf-8') as file:
        for riga in file:
            dimension,time,n_thread = riga.strip().split(";")
            if int(dimension) == dim_matrix:
                matrix_times.append(float(time))
                matrix_thread_used.append(int(n_thread))
    result = media(matrix_thread_used, matrix_times, matrix_thread_used)
    matrix_thread_used, matrix_times, inutile = zip(*result)
    counter = 0;
    for i in matrix_thread_used:
        if i == 1:
            tempo_seriale = matrix_times[counter]
        counter = counter + 1
    dati = sorted(zip(matrix_thread_used, matrix_times))
    thread_n, tempi = zip(*dati)
    for nthread, tempo in zip(matrix_thread_used, matrix_times):
        speed = tempo_seriale / tempo
        speedup.append(speed)
        efficency.append(speed/nthread*100)
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(matrix_thread_used, matrix_times, marker='o', color='b', label='Tempo (s)', linestyle='-', linewidth=2)
    plt.xlabel('Numero di Thread')
    plt.ylabel('Tempo (s)')
    plt.title('Tempo in funzione dei Thread')
    plt.grid(True)
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(matrix_thread_used, speedup, marker='o', color='g', label='Speedup', linestyle='-', linewidth=2)
    plt.xlabel('Numero di Thread')
    plt.ylabel('Speedup')
    plt.title('Speedup in funzione dei Thread')
    plt.grid(True)
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(matrix_thread_used, efficency, marker='o', color='r', label='Efficienza', linestyle='-', linewidth=2)
    plt.xlabel('Numero di Thread')
    plt.ylabel('Efficienza')
    plt.title('Efficienza in funzione dei Thread')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.show()
    file_name = f"../pdf_graph/efficency_speedup_matrix_size_{dim_matrix}.pdf"
    plt.savefig(file_name, format='pdf')
    plt.clf()


def speedup(filename, colors=None):
    #da copiare da di la in efficency
    print()
    
def efficiency(filename, colors=None):
    data = {}
    
    # Leggi i dati dal file e raccoglili in un dizionario
    with open(filename, mode='r', encoding='utf-8') as file:
        for line in file:
            values = line.strip().split(';')
            dim = int(values[0])  # Dimensione della matrice
            time = float(values[1])  # Tempo di esecuzione
            threads = int(values[2])  # Numero di thread
            
            # Aggiungi i dati al dizionario, raggruppati per dimensione e numero di thread
            if dim not in data:
                data[dim] = {}
            if threads not in data[dim]:
                data[dim][threads] = []
            data[dim][threads].append(time)

    # Calcola la media dei tempi per ciascun gruppo (dim, threads)
    averaged_data = {}

    for dim, threads_dict in data.items():
        for threads, times in threads_dict.items():
            # Calcola la media dei tempi
            average_time = sum(times) / len(times)
            if dim not in averaged_data:
                averaged_data[dim] = {}
            averaged_data[dim][threads] = average_time
    
    # Creazione del grafico
    plt.figure(figsize=(10, 6))  # Dimensione del grafico
    color_index = 0  # Indice per i colori
    default_colors = ['#ff0000', '#ff6100', '#ffdc00', '#55ff00', '#00ecff', '#0027ff', '#ae00ff', '#ff00f0', '#C70039', '#FFB6C1']
    
    # Usa i dati medi per calcolare l'efficienza e plottare
    for dim, threads_dict in averaged_data.items():
        # Ordina i dati per numero di thread
        sorted_threads = sorted(threads_dict.keys())
        sorted_times = [threads_dict[t] for t in sorted_threads]
        
        # Calcola il tempo seriale (tempo per il numero minimo di thread)
        serial_time = sorted_times[0]
        
        # Calcola speedup ed efficienza
        speedup = [serial_time / time for time in sorted_times]
        efficiency = [(s / t) * 100 for s, t in zip(speedup, sorted_threads)]  # efficienza in percentuale

        # Usa il colore fornito o il colore predefinito
        color = colors[color_index] if colors else default_colors[color_index]
        color_index = (color_index + 1) % len(default_colors)
        
        # Plotta i dati
        plt.plot(sorted_threads, efficiency, marker='o', linestyle='--', color=color, label=f'Matrice {dim}x{dim}')
    
    # Personalizzazione del grafico
    plt.xlabel('Numero di Thread')
    plt.ylabel('Efficienza (%)')
    plt.title('Efficienza in funzione del Numero di Thread per diverse Matrici')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("../pdf_graph/efficiency_matrix_sizes.pdf", format='pdf')
    # plt.show()  # Usa questa linea per visualizzare il grafico interattivamente
    plt.clf()
    
def efficiencyOMP_MPI(filename, filename2, serial_file, label1="OMP", label2="MPI", colors=None, filter_dims=[4,8,12]):
    # Leggi i file CSV
    seriale_df = pd.read_csv(serial_file, sep=';', names=['Dimensione', 'Tempo'])
    omp_df = pd.read_csv(filename, sep=';', names=['Dimensione', 'Tempo', 'Thread'])
    mpi_df = pd.read_csv(filename2, sep=';', names=['Dimensione', 'Tempo', 'Processi'])

    # Stampa le dimensioni uniche per ogni dataframe per il debug
    print("Dimensioni uniche (Seriale):", seriale_df['Dimensione'].unique())
    print("Dimensioni uniche (OMP):", omp_df['Dimensione'].unique())
    print("Dimensioni uniche (MPI):", mpi_df['Dimensione'].unique())

    # Ordina i dataframe
    seriale_df = seriale_df.sort_values(by=['Dimensione'])
    omp_df = omp_df.sort_values(by=['Dimensione', 'Thread'])
    mpi_df = mpi_df.sort_values(by=['Dimensione', 'Processi'])

    # Filtro per dimensioni specificate
    seriale_df = seriale_df[seriale_df['Dimensione'].isin(filter_dims)]
    omp_df = omp_df[omp_df['Dimensione'].isin(filter_dims)]
    mpi_df = mpi_df[mpi_df['Dimensione'].isin(filter_dims)]

    # Verifica le dimensioni dopo il filtro
    print("Dimensioni (Seriale) dopo filtro:", seriale_df['Dimensione'].unique())
    print("Dimensioni (OMP) dopo filtro:", omp_df['Dimensione'].unique())
    print("Dimensioni (MPI) dopo filtro:", mpi_df['Dimensione'].unique())

    # Calcola le medie
    seriale_mean = seriale_df.groupby('Dimensione', as_index=False)['Tempo'].mean()
    omp_mean = omp_df.groupby(['Dimensione', 'Thread'], as_index=False)['Tempo'].mean()
    mpi_mean = mpi_df.groupby(['Dimensione', 'Processi'], as_index=False)['Tempo'].mean()
    
    # Lista per raccogliere i risultati
    a = seriale_df['Dimensione'].unique();
    for dimensione in a:
        print("aaa")
        general = omp_df['Dimensione'] == dimensione
        omp_tempo_dim = general['Tempo']
        omp_thread_dim = general['Thread']
        mpi_tempo_dim = []
        mpi_thread_dim = []
        omp_speedups = []
        mpi_speedups = []
        
        tempo_seriale = omp_tempo_dim.iloc[0]
        speedup[dimensione] = [tempo_seriale/t for t in omp_tempo_dim]
        #plt.plot(omp_thread_dim, speedup, marker='o', linestyle='--', color=color, label=f'Matrice {dimensione}x{dimensione}')
        print(speedup[dimensione])
    
def media(dimensione, tempi, tipo):
    dati_raggruppati = defaultdict(list)

    # Raggruppiamo i tempi in base ai valori della dimensione e tipo
    for dim, t, tpo in zip(dimensione, tempi, tipo):
        dati_raggruppati[(dim, tpo)].append(t)

    # Ora calcoliamo la media per ciascun gruppo e salviamo i risultati
    risultati = []

    for (dim, tpo), tempi_gruppo in dati_raggruppati.items():
        media_tempi = np.mean(tempi_gruppo)  # Calcoliamo la media dei tempi per il gruppo
        risultati.append((dim, media_tempi, tpo))

    # Risultati finali
    #for r in risultati:
        #print(f"Dimensione: {r[0]}, Media tempi: {r[1]}, Tipo: {r[2]}")
    return risultati

if __name__ == "__main__":
    main()


