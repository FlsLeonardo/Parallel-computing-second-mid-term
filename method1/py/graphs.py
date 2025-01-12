import pandas as pd
import matplotlib.pyplot as plt

def speedup(filenameO,filenameS,filenameM):
    filter_dims=[4,8,12]
    
    f_o = pd.read_csv(filenameO, delimiter=";")
    f_s = pd.read_csv(filenameS, delimiter=";")
    f_m = pd.read_csv(filenameM, delimiter=";")
    
    
    f_m_avg = f_m.groupby(['Proc','Size']).agg({'Time': 'mean'}).reset_index()
    f_m_avg = f_m_avg[f_m_avg['Size'].isin(filter_dims)]
    print(f_m_avg)
    f_s_avg = f_s.groupby(['Size']).agg({'Time': 'mean'}).reset_index()
    f_s_avg = f_s_avg[f_s_avg['Size'].isin(filter_dims)]
    
    f_o_avg = f_o.groupby(['Thread', 'Size']).agg({'Time': 'mean'}).reset_index()
    f_o_avg = f_o_avg[f_o_avg['Size'].isin(filter_dims)]
    
    # Create a figure and axis for the plot
    plt.figure(figsize=(10, 6))

    #Per Speedup con OMP
    matrix_sizes = f_o_avg['Size'].unique()
    for matrix_size in matrix_sizes:
        subset = f_o_avg[f_o_avg['Size'] == matrix_size]
        omp_tempo_dim = subset['Time']
        omp_thread_dim = subset['Thread']
        seriali = f_s_avg[f_s_avg['Size'] == matrix_size]
        tempo_seriale = seriali['Time'].iloc[0]
        print(tempo_seriale)
        speedup = [tempo_seriale/t for t in omp_tempo_dim]
        print(speedup)
        plt.plot(omp_thread_dim, speedup, marker='o', linestyle='-', label=f"Matrix Size {matrix_size} OMP")
    #Per Speedup con MPI    
    for matrix_size in matrix_sizes:
        subset = f_m_avg[f_m_avg['Size'] == matrix_size]
        mpi_tempo_dim = subset['Time']
        mpi_proc_dim = subset['Proc']
        seriali = f_s_avg[f_s_avg['Size'] == matrix_size]
        tempo_seriale = seriali['Time'].iloc[0]
        speedup = [tempo_seriale/t for t in mpi_tempo_dim]
        plt.plot(mpi_proc_dim, speedup, marker='o', linestyle='-', label=f"Matrix Size {matrix_size} MPI")
        
    plt.xlabel('Process / Threads')
    plt.ylabel('Speedup')
    plt.title('differences from MPI to OMP')
    plt.grid(True)
    plt.legend()
    plt.title('Strong escaling for Speedup')
    plt.grid(True)
    plt.legend()
    # Show the plot
    file_name = f"../pdf_graph/speedup_strong_scaling.pdf"
    plt.savefig(file_name, format='pdf')
    
def efficency(filenameO,filenameS,filenameM):
    filter_dims=[4,8,12]
    
    f_o = pd.read_csv(filenameO, delimiter=";")
    f_s = pd.read_csv(filenameS, delimiter=";")
    f_m = pd.read_csv(filenameM, delimiter=";")
    
    
    f_m_avg = f_m.groupby(['Proc','Size']).agg({'Time': 'mean'}).reset_index()
    f_m_avg = f_m_avg[f_m_avg['Size'].isin(filter_dims)]
  
    f_s_avg = f_s.groupby(['Size']).agg({'Time': 'mean'}).reset_index()
    f_s_avg = f_s_avg[f_s_avg['Size'].isin(filter_dims)]
    
    f_o_avg = f_o.groupby(['Thread', 'Size']).agg({'Time': 'mean'}).reset_index()
    f_o_avg = f_o_avg[f_o_avg['Size'].isin(filter_dims)]
    
    # Create a figure and axis for the plot
    plt.figure(figsize=(10, 6))

    #Per Speedup con OMP
    matrix_sizes = f_o_avg['Size'].unique()
    for matrix_size in matrix_sizes:
        subset = f_o_avg[f_o_avg['Size'] == matrix_size]
        print(subset)
        omp_tempo_dim = subset['Time']
        omp_thread_dim = subset['Thread']
        seriali = f_s_avg[f_s_avg['Size'] == matrix_size]
        tempo_seriale = seriali['Time'].iloc[0]
      
        speedup = [tempo_seriale/t for t in omp_tempo_dim]
        efficency = [(s / t) * 100 for s, t in zip(speedup, omp_thread_dim)]
    
        plt.plot(omp_thread_dim, efficency, marker='o', linestyle='-', label=f"Matrix Size {matrix_size} OMP")
    #Per Speedup con MPI    
    for matrix_size in matrix_sizes:
        subset = f_m_avg[f_m_avg['Size'] == matrix_size]
        mpi_tempo_dim = subset['Time']
        mpi_proc_dim = subset['Proc']
        seriali = f_s_avg[f_s_avg['Size'] == matrix_size]
        tempo_seriale = seriali['Time'].iloc[0]
        speedup = [tempo_seriale/t for t in mpi_tempo_dim]
        efficency = [(s / t) * 100 for s, t in zip(speedup, mpi_proc_dim)]
        plt.plot(mpi_proc_dim, efficency, marker='o', linestyle='-', label=f"Matrix Size {matrix_size} MPI")
        
    plt.xlabel('Process / Threads')
    plt.ylabel('Efficiency')
    plt.title('differences from MPI to OMP')
    plt.grid(True)
    plt.legend()
    plt.title('Strong escaling for Efficiency')
    plt.grid(True)
    plt.legend()
    file_name = f"../pdf_graph/efficiency_strong_scaling.pdf"
    plt.savefig(file_name, format='pdf')


def speedupW(filenameO,filenameS,filenameM):
    filter_dims=[4,5,6,7] #con 2^4 devo usare 1thr con 5 devo usarne 4 con 6 devo usarne 16 con 7 devo usarne 64
    
    f_o = pd.read_csv(filenameO, delimiter=";")
    f_s = pd.read_csv(filenameS, delimiter=";")
    f_m = pd.read_csv(filenameM, delimiter=";")
    
    
    f_m_avg = f_m.groupby(['Proc','Size']).agg({'Time': 'mean'}).reset_index()
    f_m_avg = f_m_avg[f_m_avg['Size'].isin(filter_dims)]
  
    f_s_avg = f_s.groupby(['Size']).agg({'Time': 'mean'}).reset_index()
    f_s_avg = f_s_avg[f_s_avg['Size'].isin(filter_dims)]
    
    f_o_avg = f_o.groupby(['Thread', 'Size']).agg({'Time': 'mean'}).reset_index()
    f_o_avg = f_o_avg[f_o_avg['Size'].isin(filter_dims)]
    
    
    
    # Create a figure and axis for the plot
    plt.figure(figsize=(10, 6))

    #Per Speedup con OMP
    counter = 0
    speedup = []
    dim = []
    seriali = f_s_avg[f_s_avg['Size'] == 4]
    tempo_seriale = seriali[seriali['Size'] == 4]['Time'].iloc[0]
    matrix_sizes = f_o_avg['Size'].unique()
    for matrix_size in matrix_sizes:
        subset = f_o_avg[f_o_avg['Size'] == matrix_size]
        print(subset)
        omp_tempo_dim = subset['Time']
        omp_thread_dim = subset['Thread']
        if matrix_size == 4:
            time = subset[subset['Thread'] == 1]['Time'].values[0]
            speedup.append(tempo_seriale/time)
            dim.append(matrix_size)
        elif matrix_size == 5:
            time = subset[subset['Thread'] == 4]['Time'].values[0]
            speedup.append(tempo_seriale/time)
            dim.append(matrix_size)
        elif matrix_size == 6:
            time = subset[subset['Thread'] == 16]['Time'].values[0]
            speedup.append(tempo_seriale/time)
            dim.append(matrix_size)
        elif matrix_size == 7:
            time = subset[subset['Thread'] == 64]['Time'].values[0]
            speedup.append(tempo_seriale/time)
            dim.append(matrix_size)

        print(speedup)
        print(dim)
    efficency = [(s / t) * 100 for s, t in zip(speedup, dim)]
    #print(speedup)
    plt.plot(dim, efficency, marker='o', linestyle='-', label=f" OMP")
    #Per Speedup con MPI
    speedup = []
    dim = []
    seriali = f_s_avg[f_s_avg['Size'] == 4]
    tempo_seriale = seriali[seriali['Size'] == 4]['Time'].iloc[0]
    for matrix_size in matrix_sizes:
        subset = f_m_avg[f_m_avg['Size'] == matrix_size]
        mpi_tempo_dim = subset['Time']
        mpi_proc_dim = subset['Proc']
        if matrix_size == 4:
            time = subset[subset['Proc'] == 1]['Time'].values[0]
            speedup.append(tempo_seriale/time)
            dim.append(matrix_size)
        elif matrix_size == 5:
            time = subset[subset['Proc'] == 4]['Time'].values[0]
            speedup.append(tempo_seriale/time)
            dim.append(matrix_size)
        elif matrix_size == 6:
            time = subset[subset['Proc'] == 16]['Time'].values[0]
            speedup.append(tempo_seriale/time)
            dim.append(matrix_size)
        elif matrix_size == 7:
            time = subset[subset['Proc'] == 64]['Time'].values[0]
            speedup.append(tempo_seriale/time)
            dim.append(matrix_size)
        
    plt.plot(dim, speedup, marker='o', linestyle='-', label=f" MPI")
    plt.xlabel('Matrix size')
    xticks = [4, 5, 6, 7]
    xtick_labels = [f'$2^{int(i)}$' for i in xticks]
    plt.xticks(xticks, xtick_labels)
    plt.ylabel('Speedup')
    #plt.yscale("log")#--------------------looooggg
    plt.title('differences from MPI to OMP')
    plt.grid(True)
    plt.legend()
    plt.title('Weack escaling for Speedup LI CONFRONTO CON IL DATO SERIALE ANCHE SOLO CON UN PROCESSO')
    plt.grid(True)
    plt.legend()

    file_name = f"../pdf_graph/speedup_weack_scaling.pdf"
    plt.savefig(file_name, format='pdf')
    
def efficencyW(filenameO,filenameS,filenameM):
    filter_dims=[4,5,6,7] #con 2^4 devo usare 1thr con 5 devo usarne 4 con 6 devo usarne 16 con 7 devo usarne 64
    
    f_o = pd.read_csv(filenameO, delimiter=";")
    f_s = pd.read_csv(filenameS, delimiter=";")
    f_m = pd.read_csv(filenameM, delimiter=";")
    
    
    f_m_avg = f_m.groupby(['Proc','Size']).agg({'Time': 'mean'}).reset_index()
    f_m_avg = f_m_avg[f_m_avg['Size'].isin(filter_dims)]
  
    f_s_avg = f_s.groupby(['Size']).agg({'Time': 'mean'}).reset_index()
    f_s_avg = f_s_avg[f_s_avg['Size'].isin(filter_dims)]
    
    f_o_avg = f_o.groupby(['Thread', 'Size']).agg({'Time': 'mean'}).reset_index()
    f_o_avg = f_o_avg[f_o_avg['Size'].isin(filter_dims)]
    
    
    
    # Create a figure and axis for the plot
    plt.figure(figsize=(10, 6))

    #Per Speedup con OMP
    counter = 0
    speedup = []
    dim = []
    thr = []
    seriali = f_s_avg[f_s_avg['Size'] == 4]
    tempo_seriale = seriali[seriali['Size'] == 4]['Time'].iloc[0]
    matrix_sizes = f_o_avg['Size'].unique()
    for matrix_size in matrix_sizes:
        subset = f_o_avg[f_o_avg['Size'] == matrix_size]
        print(subset)
        omp_tempo_dim = subset['Time']
        omp_thread_dim = subset['Thread']
        if matrix_size == 4:
            time = subset[subset['Thread'] == 1]['Time'].values[0]
            speedup.append(tempo_seriale/time)
            dim.append(matrix_size)
            thr.append(1)
        elif matrix_size == 5:
            time = subset[subset['Thread'] == 4]['Time'].values[0]
            speedup.append(tempo_seriale/time)
            dim.append(matrix_size)
            thr.append(4)
        elif matrix_size == 6:
            time = subset[subset['Thread'] == 16]['Time'].values[0]
            speedup.append(tempo_seriale/time)
            dim.append(matrix_size)
            thr.append(16)
        elif matrix_size == 7:
            time = subset[subset['Thread'] == 64]['Time'].values[0]
            speedup.append(tempo_seriale/time)
            dim.append(matrix_size)
            thr.append(64)

        print(speedup)
        print(dim)
    efficency = [(s / t) * 100 for s, t in zip(speedup, thr)] #anche quiiiiiiiiiiiiii
    #print(speedup)
    plt.plot(dim, efficency, marker='o', linestyle='-', label=f" OMP")
    
    
    #Per Speedup con MPI
    speedup = []
    dim = []
    thr = []
    seriali = f_s_avg[f_s_avg['Size'] == 4]
    tempo_seriale = seriali[seriali['Size'] == 4]['Time'].iloc[0]
    for matrix_size in matrix_sizes:
        subset = f_m_avg[f_m_avg['Size'] == matrix_size]
        mpi_tempo_dim = subset['Time']
        mpi_proc_dim = subset['Proc']
        
        print(tempo_seriale)
        if matrix_size == 4:
            time = subset[subset['Proc'] == 1]['Time'].values[0]
            speedup.append(tempo_seriale/time)
            dim.append(matrix_size)
            thr.append(1)
        elif matrix_size == 5:
            time = subset[subset['Proc'] == 4]['Time'].values[0]
            speedup.append(tempo_seriale/time)
            dim.append(matrix_size)
            thr.append(4)
        elif matrix_size == 6:
            time = subset[subset['Proc'] == 16]['Time'].values[0]
            speedup.append(tempo_seriale/time)
            dim.append(matrix_size)
            thr.append(16)
        elif matrix_size == 7:
            time = subset[subset['Proc'] == 64]['Time'].values[0]
            speedup.append(tempo_seriale/time)
            dim.append(matrix_size)
            thr.append(64)
        
    efficency = [(s / t) * 100 for s, t in zip(speedup, thr)] #al posto della dimensione deve esserci ilnum di threadddd
    plt.plot(dim, efficency, marker='o', linestyle='-', label=f" MPI")
    plt.xlabel('Matrix size')
    xticks = [4, 5, 6, 7]
    xtick_labels = [f'$2^{int(i)}$' for i in xticks]
    plt.xticks(xticks, xtick_labels)
    plt.ylabel('Efficiency')
    plt.title('differences from MPI to OMP')
    plt.grid(True)
    plt.legend()
    plt.title('Weack escaling for Efficency LI CONFRONTO CON IL DATO SERIALE ANCHE SOLO CON UN PROCESSO')
    plt.grid(True)
    plt.legend()

    # Show the plot
    #plt.show()
    file_name = f"../pdf_graph/efficency_weack_scaling.pdf"
    plt.savefig(file_name, format='pdf')


def bandwidth(filenameO, filenameS, filenameM):
    filter_dims = [4, 5, 6, 7, 8, 9, 10, 11, 12]  # Con 2^4 -> 1 thr, 2^5 -> 4 thr, 2^6 -> 16 thr, 2^7 -> 64 thr

    # Legge i dati dai file
    f_o = pd.read_csv(filenameO, delimiter=";")
    f_s = pd.read_csv(filenameS, delimiter=";")
    f_m = pd.read_csv(filenameM, delimiter=";")
    
    float_size = 4  # bytes

    # Serial-------------------------------------------------------------------------------------------------
    bandwidths_serial = []
    sizes_serial = []
    for size in filter_dims:
        # Numero di elementi nella matrice (size^2 x size^2)
        num_elements = (2 ** size) * (2 ** size)
        
        # Calcola i byte per la matrice
        total_bytes = num_elements * float_size
        
        # Filtra dati per dimensione
        serial_data = f_s.groupby(['Size']).agg({'Time': 'mean'}).reset_index()
        serial_data = serial_data[serial_data['Size'] == size]
        serial_time = serial_data['Time'].values[0] if len(serial_data) > 0 else None
        
        if serial_time:
            serial_bandwidth =2* total_bytes / serial_time
        else:
            serial_bandwidth = None
        
        # Aggiunge i risultati alla lista
        bandwidths_serial.append(serial_bandwidth / 1e9 if serial_bandwidth else None)  # in GB/s
        sizes_serial.append(size)
    
    # OMP-------------------------------------------------------------------------------------------------
    bandwidths_omp = []
    sizes_omp = []
    for size in filter_dims:
        # Numero di elementi nella matrice (size^2 x size^2)
        num_elements = (2 ** size) * (2 ** size)
        
        # Calcola i byte per la matrice
        total_bytes = num_elements * float_size
        
        # Filtra dati per dimensione e thread
        omp_data = f_o.groupby(['Thread', 'Size']).agg({'Time': 'mean'}).reset_index()
        omp_data = omp_data[(omp_data['Size'] == size) & (omp_data['Thread'] == 64)]
        omp_time = omp_data['Time'].values[0] if len(omp_data) > 0 else None
        
        if omp_time:
            omp_bandwidth =2* total_bytes / omp_time
        else:
            omp_bandwidth = None
        
        # Aggiunge i risultati alla lista
        bandwidths_omp.append(omp_bandwidth / 1e9 if omp_bandwidth else None)  # in GB/s
        sizes_omp.append(size)

    # MPI-------------------------------------------------------------------------------------------------
    bandwidths_mpi = []
    sizes_mpi = []
    for size in filter_dims:
        # Numero di elementi nella matrice (size^2 x size^2)
        num_elements = (2 ** size) * (2 ** size)
        
        # Calcola i byte per la matrice
        total_bytes = num_elements * float_size
        
        # Filtra dati per dimensione e processi
        mpi_data = f_m.groupby(['Proc', 'Size']).agg({'Time': 'mean'}).reset_index()
        mpi_data = mpi_data[(mpi_data['Size'] == size) & (mpi_data['Proc'] == 64)]
        mpi_time = mpi_data['Time'].values[0] if len(mpi_data) > 0 else None
        
        if mpi_time:
            mpi_bandwidth = 2 * total_bytes / mpi_time
        else:
            mpi_bandwidth = None
        
        # Aggiunge i risultati alla lista
        bandwidths_mpi.append(mpi_bandwidth / 1e9 if mpi_bandwidth else None)  # in GB/s
        sizes_mpi.append(size)

    # Plotta il grafico con Serial, OMP e MPI
    plt.figure(figsize=(10, 6))
    plt.plot(sizes_serial, bandwidths_serial, marker='o', color='b', label='Serial Bandwidth')
    plt.plot(sizes_omp, bandwidths_omp, marker='x', color='r', label='OMP Bandwidth 64 Threads')
    plt.plot(sizes_mpi, bandwidths_mpi, marker='s', color='g', label='MPI Bandwidth 64 Procs')
    
    plt.xlabel('Dimensione della matrice (Size)')
    plt.ylabel('Bandwidth (GB/s)')
    plt.title('Bandwidth Seriali, OMP e MPI in Funzione della Dimensione della Matrice')
    plt.grid(True)
    plt.legend(loc='upper right')
    #plt.show()
    file_name = f"../pdf_graph/Bandwidth_Comparison.pdf"
    plt.savefig(file_name, format='pdf')

def plot_bandwidth_by_threads(filename):
    # Leggi i dati dal file CSV
    df = pd.read_csv(filename, delimiter=";")
    
    # Definisci i numeri di thread da analizzare
    thread_counts = [1, 2, 4, 8, 16, 32, 64]
    
    # Dimensione in byte per elemento (ad esempio, 4 byte per float)
    element_size = 4  # in byte
    
    # Lista per memorizzare i dati da tracciare
    bandwidth_data = []
    
    # Cicla attraverso i numeri di thread
    for threads in thread_counts:
        # Filtra i dati per il numero di thread corrente
        df_filtered = df[df['Thread'] == threads]
        
        # Calcola la bandwidth per ogni dimensione
        bandwidths = []
        sizes = []
        for size in df_filtered['Size'].unique():
            # Filtra i dati per la dimensione corrente
            df_size = df_filtered[df_filtered['Size'] == size]
            
            # Calcola il numero di elementi nella matrice (size^2)
            num_elements = (2 ** size) * (2 ** size)
            print(f"{num_elements} {size} {size}")
            # Calcola il tempo medio
            avg_time = df_size['Time'].mean()
            
            if avg_time > 0:
                # Calcola la bandwidth in GB/s
                bandwidth = 2 * (num_elements * element_size) / (avg_time)
            else:
                bandwidth = None
            
            bandwidths.append(bandwidth / 1e9 if bandwidth else None)
            sizes.append(size)
        
        # Aggiungi i dati per il numero di thread corrente
        bandwidth_data.append((sizes, bandwidths, threads))
    
    # Traccia i risultati
    plt.figure(figsize=(10, 6))
    for sizes, bandwidths, threads in bandwidth_data:
        plt.plot(sizes, bandwidths, marker='o', label=f'{threads} Threads')
    
    plt.xlabel('Dimensione della matrice (Size)')
    plt.ylabel('Bandwidth (GB/s)')
    plt.title('Bandwidth in Funzione della Dimensione della Matrice per Diversi Numeri di Thread')
    plt.grid(True)
    plt.legend()
    #plt.show()
    file_name = f"../pdf_graph/BandwidthOMP.pdf"
    plt.savefig(file_name, format='pdf')
    
def plot_bandwidth_by_proc(filename):
    # Leggi i dati dal file CSV
    df = pd.read_csv(filename, delimiter=";")
    
    # Definisci i numeri di thread da analizzare
    thread_counts = [1, 2, 4, 8, 16, 32, 64]
    
    # Dimensione in byte per elemento (ad esempio, 4 byte per float)
    element_size = 4  # in byte
    
    # Lista per memorizzare i dati da tracciare
    bandwidth_data = []
    
    # Cicla attraverso i numeri di thread
    for threads in thread_counts:
        # Filtra i dati per il numero di thread corrente
        df_filtered = df[df['Proc'] == threads]
        
        # Calcola la bandwidth per ogni dimensione
        bandwidths = []
        sizes = []
        for size in df_filtered['Size'].unique():
            # Filtra i dati per la dimensione corrente
            df_size = df_filtered[df_filtered['Size'] == size]
            
            # Calcola il numero di elementi nella matrice (size^2)
            num_elements = (2 ** size) * (2 ** size)
            print(f"{num_elements} {size} {size}")
            # Calcola il tempo medio
            avg_time = df_size['Time'].mean()
            
            if avg_time > 0:
                # Calcola la bandwidth in GB/s
                bandwidth = 2 * (num_elements * element_size) / (avg_time)
            else:
                bandwidth = None
            
            bandwidths.append(bandwidth / 1e9 if bandwidth else None)
            sizes.append(size)
        
        # Aggiungi i dati per il numero di thread corrente
        bandwidth_data.append((sizes, bandwidths, threads))
    
    # Traccia i risultati
    plt.figure(figsize=(10, 6))
    for sizes, bandwidths, threads in bandwidth_data:
        plt.plot(sizes, bandwidths, marker='o', label=f'{threads} Threads')
    
    plt.xlabel('Dimensione della matrice (Size)')
    plt.ylabel('Bandwidth (GB/s)')
    plt.title('Bandwidth in Funzione della Dimensione della Matrice per Diversi Numeri di Procs')
    plt.grid(True)
    plt.legend()
    #plt.show()
    file_name = f"../pdf_graph/BandwidthMPI.pdf"
    plt.savefig(file_name, format='pdf')

#strong escaling
efficency("../output/Omp.csv","../output/Serial.csv","../output/MPI.csv")
speedup("../output/Omp.csv","../output/Serial.csv","../output/MPI.csv")
#weack escaling
efficencyW("../output/Omp.csv","../output/Serial.csv","../output/MPI.csv")
speedupW("../output/Omp.csv","../output/Serial.csv","../output/MPI.csv")
bandwidth("../output/Omp.csv","../output/Serial.csv","../output/MPI.csv")
plot_bandwidth_by_threads("../output/Omp.csv")
plot_bandwidth_by_proc("../output/MPI.csv")