import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

def main():
    # Controllo se sono stati forniti parametri
    if len(sys.argv) < 2:
        print("Nessun parametro fornito.")
        serial("../output/Serial.csv")
        implicit("../output/Implicit.csv")
        omp("../output/Omp.csv")
        for num in range(4, 13):
            efficency_speedup("../output/Omp.csv",num)
        speedup("../output/Omp.csv")
        efficiency("../output/Omp.csv")
        sys.exit(1)
    for arg in sys.argv:
        if "--help" in arg:
            print("\nHai richiesto l'aiuto.")
            print("-type=(serial, implicit, omp)")
            print("-ES=(number)")
            sys.exit(0)
        if "-type" in arg:
            choice = sys.argv[1].split("=")[1]
            if choice not in ["serial", "implicit", "omp"]:
                print(f"Errore: Tipo '{choice}' non valido. Scegli tra 'serial', 'implicit' o 'omp'.")
                sys.exit(1)
            if choice == "serial":
                serial("../output/Serial.csv")
            elif choice == "implicit":
                implicit("../output/Implicit.csv")
            elif choice == "omp":
                omp("../output/Omp.csv")
        if "-ES" in arg: #efficency and Speedup
            matrix_dim = sys.argv[1].split("=")[1]
            efficency_speedup("../output/Omp.csv",matrix_dim)

def serial(filename):
    dimensioni = []
    tempi = []
    with open(filename, mode='r', encoding='utf-8') as file:
        for riga in file:
            valori = riga.strip().split(";")
            dimensioni.append(int(valori[0]))
            tempi.append(float(valori[1]))
        result = media(dimensioni, tempi, dimensioni)
        dimensioni, tempi, inutile = zip(*result)
        dati = sorted(zip(dimensioni, tempi))
        dimensioni, tempi = zip(*dati)
        plt.plot(dimensioni, tempi, marker='o', linestyle='-', color='r', label='Tempo medio')
        plt.xlabel('Dimensione della matrice (n)')
        plt.ylabel('Tempo di trasposizione (secondi)')
        plt.title('Tempo di Trasposizione Seriale in funzione della Dimensione della Matrice')
        plt.grid(True)
        plt.legend()
        #plt.show()
        plt.savefig("../pdf_graph/transpose_time_vs_matrix_size_Serial.pdf", format='pdf')
        plt.clf()
    
def implicit(filename, colors=None):
    # Lista di colori predefinita se non specificata
    if colors is None:
        colors = ['#ff0000', '#ff6100', '#ffdc00', '#55ff00', '#00ecff', 
                  '#0027ff', '#ae00ff', '#ff00f0', '#C70039', '#FFB6C1']

    # Dizionario per contenere i dati raggruppati per (x_value, option)
    data = defaultdict(lambda: {'x': [], 'y': []})
    
    # Lettura del file
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(';')
            if len(parts) == 3:  # Controllo formato
                x_value = int(parts[0])  # Primo valore (es. 4, 5, ...)
                y_value = float(parts[1])  # Secondo valore (es. 2.87592e-07, ...)
                option = parts[2]  # Terzo valore (es. O1, O2, ...)
                
                # Aggiunta dei dati al dizionario, raggruppati per x_value e option
                data[(x_value, option)]['x'].append(x_value)
                data[(x_value, option)]['y'].append(y_value)
    
    # Calcolare la media dei tempi di esecuzione per ogni combinazione di (x_value, option)
    averaged_data = defaultdict(lambda: {'x': [], 'y': []})
    for (x_value, option), values in data.items():
        # Media dei valori 'y' per ogni (x_value, option)
        avg_y = sum(values['y']) / len(values['y'])
        averaged_data[option]['x'].append(x_value)
        averaged_data[option]['y'].append(avg_y)
    
    # Creazione del grafico
    plt.figure(figsize=(10, 6))
    color_index = 0  # Indice per i colori

    for option, values in averaged_data.items():
        # Colore ciclico per le linee
        color = colors[color_index % len(colors)]
        plt.plot(values['x'], values['y'], marker='o',linestyle='--', label=option, color=color)
        color_index += 1  # Incrementa l'indice del colore
    
    # Personalizzazione del grafico
    plt.xlabel('Matrix Size', fontsize=14)
    plt.ylabel('Execution Time (Average)', fontsize=14)
    plt.title('Performance by Compiler Options (Averaged)', fontsize=16)
    plt.legend(title="Options")
    plt.grid(True)
    plt.tight_layout()
    
    # Salvataggio del grafico
    plt.savefig("../pdf_graph/transpose_time_vs_matrix_size_Implicit.pdf", format='pdf')
    plt.clf()


def omp(filename):
    dimensioni = []
    tempi = []
    thread_n = []
    with open(filename, mode='r', encoding='utf-8') as file:
        for riga in file:
            valori = riga.strip().split(";")
            dimensioni.append(int(valori[0]))
            tempi.append(float(valori[1]))
            thread_n.append(str(valori[2]))
    result = media(dimensioni, tempi, thread_n)
    dimensioni, tempi, thread_n = zip(*result)
    dati = sorted(zip(thread_n, dimensioni, tempi))
    thread_n, dimensioni, tempi = zip(*dati)
    dim = [[],[],[],[],[],[],[],[]]                         #dimensione divisa in base ai num threads
    time = [[],[],[],[],[],[],[],[]]                          #tempo diviso in base ai num threads
    for nthread, dim_, tempo in zip(thread_n, dimensioni, tempi):
        if nthread == "1":
            time[0].append(tempo)
            dim[0].append(dim_)
        elif nthread == "2":
            time[1].append(tempo)
            dim[1].append(dim_)
        elif nthread == "4":
            time[2].append(tempo)
            dim[2].append(dim_)
        elif nthread == "8":
            time[3].append(tempo)
            dim[3].append(dim_)
        elif nthread == "16":
            time[4].append(tempo)
            dim[4].append(dim_)
        elif nthread == "32":
            time[5].append(tempo)
            dim[5].append(dim_)
        elif nthread == "64":
            time[6].append(tempo)
            dim[6].append(dim_)
        elif nthread == "96":
            time[7].append(tempo)
            dim[7].append(dim_)  
    plt.plot(dim[0], time[0], marker='o', linestyle='--', color='#ff0000', label='1 thread')
    plt.plot(dim[1], time[1], marker='o', linestyle='--', color='#ff6100', label='2 thread')
    plt.plot(dim[2], time[2], marker='o', linestyle='--', color='#ffdc00', label='4 thread')
    plt.plot(dim[3], time[3], marker='o', linestyle='--', color='#55ff00', label='8 thread')
    plt.plot(dim[4], time[4], marker='o', linestyle='--', color='#00ecff', label='16 thread')
    plt.plot(dim[5], time[5], marker='o', linestyle='--', color='#0027ff', label='32 thread')
    plt.plot(dim[6], time[6], marker='o', linestyle='--', color='#ae00ff', label='64 thread')
    plt.plot(dim[7], time[7], marker='o', linestyle='--', color='#ff00f0', label='96 thread')
    plt.yscale("log")
    plt.xlabel('Dimensione della matrice (n)')
    plt.ylabel('Tempo di trasposizione (secondi)')
    plt.title('Tempo di Trasposizione Omp in funzione della Dimensione della Matrice')
    plt.grid(True)
    plt.legend()
    #plt.show()
    plt.savefig("../pdf_graph/transpose_time_vs_matrix_size_Omp.pdf", format='pdf')
    plt.clf()
  
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
    data = {}
    
    # Leggi i dati dal file e raccoglili in un dizionario
    with open(filename, mode='r', encoding='utf-8') as file:
        for line in file:
            values = line.strip().split(';')
            dim = int(values[0])  # Dimensione della matrice
            time = float(values[1])  # Tempo di esecuzione
            threads = int(values[2])  # Numero di thread
            
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
    
    # Usa i dati medi per calcolare lo speedup e plottare
    for dim, threads_dict in averaged_data.items():
        # Ordina i dati per numero di thread
        sorted_threads = sorted(threads_dict.keys())
        sorted_times = [threads_dict[t] for t in sorted_threads]
        
        # Calcola il tempo seriale (tempo per il numero minimo di thread)
        serial_time = sorted_times[0]
        
        # Calcola speedup
        speedup = [serial_time / time for time in sorted_times]
        
        # Se sono stati forniti dei colori personalizzati
        color = colors[color_index] if colors else default_colors[color_index]
        color_index = (color_index + 1) % len(default_colors)
        
        # Plotta i dati
        plt.plot(sorted_threads, speedup, marker='o', linestyle='--', color=color, label=f'Matrice {dim}x{dim}')
    
    # Personalizzazione del grafico
    plt.xlabel('Numero di Thread')
    plt.ylabel('Speedup')
    plt.title('Speedup in funzione del Numero di Thread per diverse Matrici')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("../pdf_graph/speedup_matrix_sizes.pdf", format='pdf')
    # plt.show()  # Usa questa linea per visualizzare il grafico interattivamente
    plt.clf()
    
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
    for r in risultati:
        print(f"Dimensione: {r[0]}, Media tempi: {r[1]}, Tipo: {r[2]}")
    return risultati
if __name__ == "__main__":
    main()
