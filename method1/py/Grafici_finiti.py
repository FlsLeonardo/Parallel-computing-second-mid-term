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
        
        print(tempo_seriale)
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
        
    efficency = [(s / t) * 100 for s, t in zip(speedup, dim)]
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


#strong escaling
efficency("../output/Omp.csv","../output/Serial.csv","../output/MPI.csv")
speedup("../output/Omp.csv","../output/Serial.csv","../output/MPI.csv")
#weack escaling
efficencyW("../output/Omp.csv","../output/Serial.csv","../output/MPI.csv")
speedupW("../output/Omp.csv","../output/Serial.csv","../output/MPI.csv")